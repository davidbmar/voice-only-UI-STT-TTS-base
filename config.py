"""Minimal voice pipeline settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    llm_provider: str = "claude"
    host: str = "127.0.0.1"
    port: int = 8080
    ice_servers_json: str = '[{"urls":"stun:stun.l.google.com:19302"}]'
    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

runtime_settings = {
    "barge_in_enabled": True,
    "tts_voice": "",
    "tts_engine": "piper",
    "vad_energy_threshold": 300,
    "vad_speech_confirm_frames": 1,
    "vad_silence_gap": 8,
    "barge_in_energy_threshold": 600,
    "barge_in_confirm_frames": 2,
    "stt_model_size": "base",
}

model_status = {
    "stt": "not_loaded",       # not_loaded | loading | ready
    "stt_model": "base",
    "tts": "not_loaded",       # not_loaded | loading | ready
    "tts_voice": "",
}
