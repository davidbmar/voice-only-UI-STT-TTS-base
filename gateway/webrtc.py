"""WebRTC session management — PeerConnection lifecycle and ICE config."""

import asyncio
import json
import logging
import os
import re

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

from engine.adapter import create_generator
from engine.types import AudioChunk
from gateway.audio.audio_queue import AudioQueue
from gateway.audio.webrtc_audio_source import WebRTCAudioSource

FRAME_SAMPLES = 960  # 20ms at 48kHz
SAMPLE_RATE = 48000

log = logging.getLogger("webrtc")


def ice_servers_to_rtc(servers: list) -> list:
    """Convert ICE server dicts to RTCIceServer objects."""
    result = []
    for s in servers:
        urls = s.get("urls", s.get("url", ""))
        if isinstance(urls, str):
            urls = [urls]
        result.append(RTCIceServer(
            urls=urls,
            username=s.get("username", ""),
            credential=s.get("credential", ""),
        ))
    return result


class QueuedGenerator:
    """Reads PCM from an AudioQueue FIFO in 20ms chunks.

    Same interface as SineWaveGenerator (next_chunk() → AudioChunk)
    so WebRTCAudioSource doesn't need to change.
    """

    def __init__(self, queue: AudioQueue):
        self.queue = queue

    def next_chunk(self) -> AudioChunk:
        """Read one 20ms frame (960 samples = 1920 bytes) from the queue."""
        pcm = self.queue.read(FRAME_SAMPLES * 2)  # 2 bytes per int16 sample
        return AudioChunk(samples=pcm, sample_rate=SAMPLE_RATE, channels=1)


class Session:
    """Manages one WebRTC peer connection and its audio track."""

    def __init__(self, ice_servers: list = None):
        rtc_servers = ice_servers_to_rtc(ice_servers or [])
        config = RTCConfiguration(iceServers=rtc_servers) if rtc_servers else RTCConfiguration()
        self._pc = RTCPeerConnection(configuration=config)
        self._audio_source = WebRTCAudioSource()
        self._generator = None

        # FIFO audio queue for TTS — never drops audio, drains sentence by sentence
        self._audio_queue = AudioQueue()
        self._tts_generator = QueuedGenerator(self._audio_queue)

        # Mic recording state
        self._recording = False
        self._mic_frames: list[bytes] = []
        self._mic_track = None
        self._mic_recv_task: asyncio.Task | None = None
        self._transcribe_task: asyncio.Task | None = None
        self._on_transcription = None  # callback for partial results
        self._transcribe_interval = 5  # seconds between partial transcriptions

        # Log state changes
        @self._pc.on("connectionstatechange")
        async def on_conn_state():
            log.info("Connection state: %s", self._pc.connectionState)

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_state():
            log.info("ICE connection state: %s", self._pc.iceConnectionState)

        @self._pc.on("icegatheringstatechange")
        async def on_ice_gather():
            log.info("ICE gathering state: %s", self._pc.iceGatheringState)

        @self._pc.on("track")
        async def on_track(track):
            if track.kind != "audio":
                return
            log.info("Received remote audio track from browser mic")
            self._mic_track = track
            self._mic_recv_task = asyncio.ensure_future(self._recv_mic_audio(track))

    async def handle_offer(self, sdp: str) -> str:
        """Process client SDP offer, return SDP answer.

        aiortc bundles all ICE candidates into the answer SDP
        automatically (no trickle ICE support).
        """
        # Add our audio track to the connection
        self._pc.addTrack(self._audio_source)

        # Set the remote offer
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self._pc.setRemoteDescription(offer)

        # Create and set local answer
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        log.info("SDP answer created")
        return self._pc.localDescription.sdp

    def start_audio(self, voice_id: str):
        """Start streaming audio for the given voice."""
        self._generator = create_generator(voice_id)
        self._audio_source.set_generator(self._generator)
        log.info("Audio started: %s", voice_id)

    def stop_audio(self):
        """Stop streaming audio (track sends silence)."""
        self._audio_source.clear_generator()
        self._generator = None
        log.info("Audio stopped")

    def stop_speaking(self):
        """Stop TTS playback — clear the audio queue and detach generator."""
        self._audio_queue.clear()
        self._audio_source.clear_generator()
        log.info("TTS playback stopped, queue cleared")

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Strip markdown formatting so TTS reads clean prose.

        LLMs return **bold**, *italic*, bullet lists, etc. that Piper
        would speak as literal "asterisk asterisk". This converts
        markdown to plain speech-friendly text.
        """
        # Remove markdown headers: ## Header → Header
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Convert bullet points BEFORE bold (so * bullets don't confuse ** bold)
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
        # Convert numbered lists: "1. item" → "item"
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic markers: **text** → text, *text* → text
        text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
        # Catch any remaining stray asterisks
        text = re.sub(r'\*{1,3}', '', text)
        # Remove markdown links: [text](url) → text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Track if URLs were present, then strip them along with surrounding phrases
        has_urls = bool(re.search(r'https?://\S+', text))
        # Strip "Visit/Check/See URL" phrases and bare URLs
        text = re.sub(r'(?:visit|check|see|at|on|from)\s+https?://\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'https?://\S+', '', text)
        # Remove inline code backticks: `code` → code
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Collapse multiple newlines into sentence breaks
        text = re.sub(r'\n{2,}', '. ', text)
        # Single newlines to spaces
        text = re.sub(r'\n', ' ', text)
        # Clean up multiple spaces/periods
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = text.strip()
        # If URLs were stripped, add a spoken note about the links
        if has_urls and text:
            text += " See the links on screen for more details."
        return text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for incremental TTS."""
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p for p in parts if p.strip()]

    async def speak_text(self, text: str, voice_id: str = "") -> float:
        """Run TTS sentence-by-sentence and enqueue audio into the FIFO.

        Each sentence is synthesized in a thread, then enqueued immediately.
        The WebRTC track starts playing the first sentence while later
        sentences are still being synthesized.

        Returns total playback duration in seconds.
        """
        from engine.tts import synthesize

        # Ensure the TTS generator is attached to the audio source
        self._audio_source.set_generator(self._tts_generator)

        text = self._clean_for_speech(text)
        sentences = self._split_sentences(text)
        log.info("TTS: %d sentences to synthesize", len(sentences))

        loop = asyncio.get_event_loop()
        total_bytes = 0
        for i, sentence in enumerate(sentences):
            pcm_48k = await loop.run_in_executor(None, synthesize, sentence, voice_id)
            if pcm_48k:
                self._audio_queue.enqueue(pcm_48k)
                total_bytes += len(pcm_48k)
                log.debug("TTS sentence %d/%d enqueued: %d bytes — %r",
                          i + 1, len(sentences), len(pcm_48k), sentence[:60])

        duration = total_bytes / (SAMPLE_RATE * 2)  # 48kHz int16
        log.info("TTS total: %d bytes, %.1fs playback", total_bytes, duration)
        return duration

    async def _recv_mic_audio(self, track):
        """Background task: continuously receive audio frames from the browser mic track."""
        logged_format = False
        while True:
            try:
                frame = await track.recv()
            except Exception:
                log.info("Mic track ended")
                break

            if not logged_format:
                arr = frame.to_ndarray()
                log.info(
                    "Mic frame format=%s rate=%d samples=%d shape=%s dtype=%s range=[%s, %s]",
                    frame.format.name, frame.sample_rate, frame.samples,
                    arr.shape, arr.dtype, arr.min(), arr.max(),
                )
                logged_format = True

            if self._recording:
                arr = frame.to_ndarray()
                # Handle float formats (fltp/flt) — scale to int16
                if arr.dtype in (np.float32, np.float64):
                    arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
                # s16 interleaved stereo: shape=(1, samples*channels) — take every other sample for mono
                flat = arr.flatten()
                channels = flat.shape[0] // frame.samples
                if channels > 1:
                    flat = flat[::channels]  # take left channel
                pcm = flat.astype(np.int16).tobytes()
                self._mic_frames.append(pcm)

    def start_recording(self, on_transcription=None):
        """Start buffering incoming mic audio frames.

        Args:
            on_transcription: async callback(text, partial) called with
                              partial transcriptions every TRANSCRIBE_INTERVAL seconds.
        """
        self._mic_frames.clear()
        self._recording = True
        self._on_transcription = on_transcription
        self._transcribe_task = asyncio.ensure_future(self._periodic_transcribe())
        log.info("Mic recording started (live transcription enabled)")

    async def _periodic_transcribe(self):
        """Background task: transcribe accumulated audio every N seconds."""
        from engine.stt import transcribe

        interval = self._transcribe_interval
        loop = asyncio.get_event_loop()

        while self._recording:
            await asyncio.sleep(interval)
            if not self._recording:
                break
            if not self._mic_frames:
                continue

            # Snapshot all audio accumulated so far (don't clear — rolling full)
            pcm_data = b"".join(self._mic_frames)
            log.debug("Partial transcription: %d frames, %d bytes", len(self._mic_frames), len(pcm_data))

            text, _, _ = await loop.run_in_executor(None, transcribe, pcm_data, SAMPLE_RATE)

            if text and self._on_transcription and self._recording:
                await self._on_transcription(text, True)

    async def stop_recording(self) -> str:
        """Stop recording, cancel periodic task, do final transcription."""
        self._recording = False

        # Cancel periodic transcription
        if self._transcribe_task:
            self._transcribe_task.cancel()
            self._transcribe_task = None

        if not self._mic_frames:
            log.warning("No mic frames captured")
            return "", 0.0, 0.0, 0.0

        # Final transcription of all audio
        pcm_data = b"".join(self._mic_frames)
        num_frames = len(self._mic_frames)
        self._mic_frames.clear()

        audio_duration_s = len(pcm_data) / (SAMPLE_RATE * 2)  # 2 bytes per int16 sample
        log.info("Mic recording stopped: %d frames, %d bytes, %.2fs", num_frames, len(pcm_data), audio_duration_s)

        from engine.stt import transcribe
        loop = asyncio.get_event_loop()
        text, no_speech_prob, avg_logprob = await loop.run_in_executor(None, transcribe, pcm_data, SAMPLE_RATE)
        return text, no_speech_prob, avg_logprob, audio_duration_s

    async def close(self):
        """Tear down the peer connection."""
        self.stop_audio()
        self._recording = False
        if self._transcribe_task:
            self._transcribe_task.cancel()
        if self._mic_recv_task:
            self._mic_recv_task.cancel()
        await self._pc.close()
        log.info("Session closed")
