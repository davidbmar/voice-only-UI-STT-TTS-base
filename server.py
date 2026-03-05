"""Minimal FastAPI voice server."""

import logging
import os
import signal
import struct
import subprocess

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
)


def _kill_stale_server(port: int) -> None:
    try:
        pids = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        for pid in pids.splitlines():
            pid = int(pid)
            if pid != os.getpid():
                os.kill(pid, signal.SIGTERM)
    except (subprocess.CalledProcessError, OSError):
        pass


app = FastAPI(title="Voice Base")


@app.get("/")
async def index():
    return FileResponse("web/index.html")


@app.get("/admin")
async def admin_page():
    return FileResponse("web/admin.html")


# ── Config API ─────────────────────────────────────────────

WRITABLE_KEYS = {
    "barge_in_enabled", "tts_voice", "tts_engine",
    "vad_energy_threshold", "vad_speech_confirm_frames", "vad_silence_gap",
    "barge_in_energy_threshold", "barge_in_confirm_frames", "stt_model_size",
}


@app.get("/api/config")
async def get_config():
    from config import model_status, runtime_settings
    return JSONResponse({**runtime_settings, **{"model_status": model_status}})


@app.post("/api/config")
async def update_config(request: Request):
    from config import model_status, runtime_settings
    body = await request.json()
    for key, value in body.items():
        if key in WRITABLE_KEYS:
            runtime_settings[key] = value
    return JSONResponse({**runtime_settings, **{"model_status": model_status}})


# ── Voice / TTS API ────────────────────────────────────────

@app.get("/api/voices")
async def get_voices():
    from engine.tts import list_voices
    return JSONResponse(list_voices())


@app.post("/api/tts/preview")
async def tts_preview(request: Request):
    import asyncio
    body = await request.json()
    text = body.get("text", "").strip()
    voice = body.get("voice", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    try:
        from engine.tts import synthesize
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(None, synthesize, text, voice)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # Build WAV in-memory: 48kHz mono int16
    sample_rate = 48000
    num_channels = 1
    bits_per_sample = 16
    data_size = len(pcm)
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1,  # PCM format
        num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return Response(content=wav_header + pcm, media_type="audio/wav")


@app.post("/api/tts/download")
async def tts_download(request: Request):
    import asyncio
    body = await request.json()
    voice_id = body.get("voice", "")
    if not voice_id:
        return JSONResponse({"error": "No voice specified"}, status_code=400)

    try:
        from engine.tts import _download_model, _CATALOG_BY_ID
        if voice_id not in _CATALOG_BY_ID:
            return JSONResponse({"error": f"Unknown voice: {voice_id}"}, status_code=400)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _download_model, voice_id)
        return JSONResponse({"status": "downloaded", "voice": voice_id})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── STT API ────────────────────────────────────────────────

@app.get("/api/stt/status")
async def stt_status():
    from engine.stt import get_status
    return JSONResponse(get_status())


@app.post("/api/stt/switch")
async def stt_switch(request: Request):
    from engine.stt import switch_model, get_status, VALID_SIZES
    body = await request.json()
    size = body.get("size", "")
    if size not in VALID_SIZES:
        return JSONResponse({"error": f"Invalid size, must be one of {VALID_SIZES}"}, status_code=400)
    switch_model(size)
    return JSONResponse(get_status())


@app.post("/api/stt/download")
async def stt_download(request: Request):
    import asyncio
    from engine.stt import download_model, get_status, VALID_SIZES
    body = await request.json()
    size = body.get("size", "")
    if size not in VALID_SIZES:
        return JSONResponse({"error": f"Invalid size, must be one of {VALID_SIZES}"}, status_code=400)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, download_model, size)
        return JSONResponse(get_status())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── WebSocket ──────────────────────────────────────────────

@app.websocket("/ws")
async def voice_ws(websocket: WebSocket):
    from gateway.server import handle_signaling_ws
    await handle_signaling_ws(websocket)


# Static files must be mounted LAST (catch-all)
app.mount("/static", StaticFiles(directory="web"), name="static")


if __name__ == "__main__":
    import uvicorn
    from config import settings

    _kill_stale_server(settings.port)
    uvicorn.run("server:app", host=settings.host, port=settings.port, reload=True)
