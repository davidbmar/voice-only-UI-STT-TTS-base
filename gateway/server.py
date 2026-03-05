"""WebSocket signaling server for WebRTC voice calls.

Simplified voice pipeline: mic → VAD → STT → LLM → TTS → speaker.
No FSM, no workflows — just a voice conversation with Claude.

Protocol messages (Client → Server):
  {"type": "hello"}                      → request ICE servers
  {"type": "webrtc_offer", "sdp": "..."}  → send SDP offer
  {"type": "hangup"}                     → end call
  {"type": "ping"}                       → keepalive

Server → Client:
  {"type": "hello_ack", "ice_servers": [...]}
  {"type": "webrtc_answer", "sdp": "..."}
  {"type": "transcript", "text": "..."}   → user speech displayed in chat
  {"type": "response", "text": "..."}     → assistant response displayed in chat
  {"type": "status", "text": "..."}       → status updates (listening, thinking, etc.)
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from config import model_status, settings, runtime_settings
from gateway.turn import fetch_twilio_turn_credentials

log = logging.getLogger("gateway.server")


def _get_fallback_ice_servers() -> list:
    """Parse the fallback ICE servers from settings."""
    try:
        return json.loads(settings.ice_servers_json)
    except (json.JSONDecodeError, TypeError):
        log.warning("Invalid ICE_SERVERS_JSON, using empty list")
        return []


def _compute_rms(pcm_bytes: bytes) -> float:
    """Compute RMS energy of int16 PCM audio."""
    import numpy as np

    if len(pcm_bytes) < 2:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples**2)))


async def _transcribe_webrtc(pcm_48k: bytes, ws: WebSocket | None = None) -> str:
    """Transcribe 48kHz int16 PCM using faster-whisper."""
    try:
        from engine.stt import transcribe

        # Send loading status on first model load
        if model_status["stt"] != "ready" and ws:
            await _send_ws(ws, {"type": "status", "text": "Loading STT model..."})

        loop = asyncio.get_event_loop()
        text, no_speech_prob, _ = await loop.run_in_executor(
            None, transcribe, pcm_48k, 48000
        )
        if no_speech_prob > 0.6:
            return ""
        return text
    except ImportError:
        log.warning("STT not available (faster-whisper not installed)")
        return ""


def _get_llm_response(text: str, history: list) -> str:
    """Simple Claude call — no FSM, no workflows, just conversation.

    This is a sync function — run via run_in_executor from async code.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    history.append({"role": "user", "content": text})
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system="You are a helpful voice assistant in a real-time voice conversation. The user is speaking to you through a microphone — their speech is transcribed via Whisper and sent to you as text. You CAN hear them. Keep responses concise (1-2 sentences) since they will be spoken aloud via text-to-speech.",
        messages=list(history),
    )
    reply = response.content[0].text
    history.append({"role": "assistant", "content": reply})
    return reply


async def _speak(session, text: str, ws: WebSocket | None = None) -> float:
    """Synthesize and speak text via piper TTS. Returns playback duration."""
    voice = runtime_settings.get("tts_voice", "") or "en_US-lessac-medium"
    try:
        if model_status["tts"] != "ready" and ws:
            await _send_ws(ws, {"type": "status", "text": "Loading TTS voice..."})
        duration = await session.speak_text(text, voice)
        model_status["tts"] = "ready"
        model_status["tts_voice"] = voice
        return duration or 0.0
    except Exception as e:
        log.error("TTS failed: %s", e)
        return 0.0


async def _wait_for_playback(session, duration: float) -> bool:
    """Wait for TTS playback, detecting barge-in.

    Returns True if interrupted by user speech.
    """
    POLL_INTERVAL = 0.1
    BARGE_IN_THRESHOLD = runtime_settings.get("barge_in_energy_threshold", 600)
    BARGE_IN_CONFIRM = runtime_settings.get("barge_in_confirm_frames", 2)
    FRAMES_PER_POLL = 5

    elapsed = 0.0
    barge_confirm = 0

    while elapsed < duration:
        await asyncio.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        if not runtime_settings.get("barge_in_enabled", False):
            continue

        if session._mic_frames:
            recent_frames = session._mic_frames[-FRAMES_PER_POLL:]
            rms_values = [_compute_rms(f) for f in recent_frames]
            rms = max(rms_values) if rms_values else 0.0

            if rms >= BARGE_IN_THRESHOLD:
                barge_confirm += 1
                if barge_confirm >= BARGE_IN_CONFIRM:
                    log.info("Barge-in detected (rms=%.0f), stopping TTS", rms)
                    session.stop_speaking()
                    frames_to_keep = (barge_confirm + 1) * FRAMES_PER_POLL
                    session._mic_frames = session._mic_frames[-frames_to_keep:]
                    return True
            else:
                barge_confirm = 0

    session._mic_frames.clear()
    return False


async def _send_ws(ws: WebSocket, msg: dict):
    """Send JSON over WebSocket, ignoring errors if disconnected."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass


async def _voice_loop(session, ws: WebSocket) -> None:
    """Main voice loop: greeting → listen → transcribe → respond → repeat."""
    conversation_history = []

    # 1. Greeting
    greeting = "Hello! I'm your voice assistant. How can I help you?"
    await _send_ws(ws, {"type": "response", "text": greeting})
    duration = await _speak(session, greeting, ws)

    # 2. Enable mic capture
    session._recording = True
    session._mic_frames.clear()

    # Wait for greeting playback
    interrupted = await _wait_for_playback(session, duration)
    await _send_ws(ws, {"type": "status", "text": "Listening..."})

    # 3. Wait for mic track
    for i in range(50):
        if session._mic_track is not None:
            break
        await asyncio.sleep(0.1)
    if session._mic_track is None:
        log.error("No mic track received after 5s")
        return

    # 4. VAD loop
    MIN_AUDIO = 9600
    MAX_BUFFER = 48000 * 2 * 30
    silence_count = 0
    speech_confirm_count = 0
    has_speech = interrupted
    done = False

    while not done:
        await asyncio.sleep(0.1)

        # Check connection state
        pc_state = getattr(session._pc, "connectionState", None)
        if pc_state in {"closed", "failed", "disconnected"}:
            log.info("WebRTC connection %s — ending", pc_state)
            break

        if not session._mic_frames:
            continue

        threshold = runtime_settings.get("vad_energy_threshold", 300)
        confirm_needed = runtime_settings.get("vad_speech_confirm_frames", 1)
        silence_gap = runtime_settings.get("vad_silence_gap", 8)

        latest_frame = session._mic_frames[-1]
        rms = _compute_rms(latest_frame)

        if rms >= threshold:
            speech_confirm_count += 1
            if not has_speech and speech_confirm_count >= confirm_needed:
                log.info("Speech detected (rms=%.0f)", rms)
                has_speech = True
            silence_count = 0
        else:
            speech_confirm_count = 0
            silence_count += 1

        total = sum(len(f) for f in session._mic_frames)

        # Silence after speech → process
        if has_speech and silence_count >= silence_gap and total > MIN_AUDIO:
            pcm = b"".join(session._mic_frames)
            session._mic_frames.clear()
            has_speech = False
            silence_count = 0
            speech_confirm_count = 0

            await _send_ws(ws, {"type": "status", "text": "Transcribing..."})
            text = await _transcribe_webrtc(pcm, ws)
            log.info("Transcription: %r", text)

            if text and text.strip():
                await _send_ws(ws, {"type": "transcript", "text": text})
                await _send_ws(ws, {"type": "status", "text": "Thinking..."})

                response = await asyncio.get_event_loop().run_in_executor(
                    None, _get_llm_response, text, conversation_history
                )

                log.info("Response: %s", response[:100])
                await _send_ws(ws, {"type": "response", "text": response})

                duration = await _speak(session, response, ws)
                interrupted = await _wait_for_playback(session, duration)
                if interrupted:
                    has_speech = True
                await _send_ws(ws, {"type": "status", "text": "Listening..."})

        elif total > MAX_BUFFER:
            log.warning("Buffer overflow (%d bytes), clearing", total)
            session._mic_frames.clear()
            has_speech = False
            silence_count = 0
            speech_confirm_count = 0


async def handle_signaling_ws(ws: WebSocket) -> None:
    """Handle one WebRTC signaling WebSocket connection."""
    await ws.accept()
    log.info("Signaling WebSocket connected")

    session = None
    voice_task: asyncio.Task | None = None
    ice_servers: list = []

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "hello":
                ice_servers = await fetch_twilio_turn_credentials()
                if not ice_servers:
                    ice_servers = _get_fallback_ice_servers()

                await ws.send_json({
                    "type": "hello_ack",
                    "ice_servers": ice_servers,
                })
                log.info("Sent %d ICE servers", len(ice_servers))

            elif msg_type == "webrtc_offer":
                sdp = msg.get("sdp", "")
                if not sdp:
                    await ws.send_json({"type": "error", "message": "Missing SDP"})
                    continue

                try:
                    from gateway.webrtc import Session

                    session = Session(ice_servers=ice_servers)
                    answer_sdp = await session.handle_offer(sdp)

                    await ws.send_json({
                        "type": "webrtc_answer",
                        "sdp": answer_sdp,
                    })
                    log.info("WebRTC session created")

                    voice_task = asyncio.ensure_future(
                        _voice_loop(session, ws)
                    )
                except ImportError as e:
                    log.error("WebRTC not available: %s", e)
                    await ws.send_json({
                        "type": "error",
                        "message": "WebRTC not available. Install: pip install aiortc av",
                    })
                except Exception as e:
                    log.error("WebRTC setup failed: %s", e)
                    await ws.send_json({
                        "type": "error",
                        "message": f"WebRTC setup failed: {e}",
                    })

            elif msg_type == "hangup":
                log.info("Client hangup")
                if voice_task and not voice_task.done():
                    voice_task.cancel()
                    voice_task = None
                if session:
                    await session.close()
                    session = None

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

            else:
                await ws.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        log.info("Signaling WebSocket disconnected")
    except Exception as e:
        log.error("Signaling error: %s", e)
    finally:
        if voice_task and not voice_task.done():
            voice_task.cancel()
        if session:
            await session.close()
            log.info("Session cleaned up")
