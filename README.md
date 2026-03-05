# Voice Base — Standalone Voice UI

Minimal browser-to-server voice pipeline:
**Browser mic → WebRTC → STT → Claude LLM → TTS → Browser speaker**

No FSM, no workflows — just voice conversation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Run

```bash
source .venv/bin/activate
uvicorn server:app --port 8081
```

Open http://localhost:8081 → click **Call** → speak → see transcript + hear response.

## Admin Panel

Visit http://localhost:8081/admin to configure the voice pipeline at runtime:

- **VAD Settings** — Energy threshold, speech confirm frames, silence gap
- **Barge-in** — Toggle + threshold/confirm tuning for interrupting TTS
- **STT Model** — Switch between whisper sizes (tiny/base/small/medium), pre-download models
- **Voice Selection** — Browse piper voices, download on click, preview with text-to-speech
- **Live Config** — See current runtime_settings JSON, updates in real-time

All changes take effect immediately — no server restart needed.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/admin` | GET | Admin panel UI |
| `/api/config` | GET | Current runtime settings + model status |
| `/api/config` | POST | Update whitelisted config keys |
| `/api/voices` | GET | Piper voice list with download status |
| `/api/tts/preview` | POST | Synthesize text → WAV audio blob |
| `/api/tts/download` | POST | Pre-download a piper voice |
| `/api/stt/status` | GET | STT model load state |
| `/api/stt/switch` | POST | Switch STT model size |
| `/api/stt/download` | POST | Pre-download a whisper model |

## Architecture

```
Browser mic → WebRTC (aiortc) → VAD (energy) → STT (faster-whisper)
    → Claude API → TTS (piper) → WebRTC audio → Browser speaker
```

All text is relayed over WebSocket so the browser shows live transcripts.
Model loading status ("Loading STT model...", "Loading TTS voice...") is sent to the browser on first use.

## Files

- `server.py` — FastAPI entry point + API endpoints
- `config.py` — Settings (loads from .env), runtime_settings, model_status
- `gateway/server.py` — WebSocket signaling + VAD voice loop + loading status
- `gateway/webrtc.py` — WebRTC session (PeerConnection, mic, TTS playback)
- `gateway/turn.py` — Twilio TURN credentials (optional)
- `engine/stt.py` — faster-whisper STT with model switching + download
- `engine/tts.py` — Piper TTS with voice catalog + download
- `web/index.html` — Browser UI (voice call)
- `web/admin.html` — Admin panel (config, models, voices)
