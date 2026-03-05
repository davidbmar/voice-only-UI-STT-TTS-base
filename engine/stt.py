"""Faster-Whisper STT wrapper — audio bytes to text."""

import logging

import numpy as np
from scipy.signal import resample

from config import model_status, runtime_settings

log = logging.getLogger("stt")

VALID_SIZES = ("tiny", "base", "small", "medium")

# Lazy-loaded Whisper model
_model = None
_loaded_size: str | None = None


def _get_model():
    """Load the faster-whisper model on first use (auto-downloads).

    Reads desired size from runtime_settings and reloads if it changed.
    """
    global _model, _loaded_size

    desired = runtime_settings.get("stt_model_size", "base")
    if _model is not None and _loaded_size == desired:
        return _model

    from faster_whisper import WhisperModel

    model_status["stt"] = "loading"
    model_status["stt_model"] = desired
    log.info("Loading faster-whisper model: %s ...", desired)

    _model = WhisperModel(desired, device="cpu", compute_type="int8")
    _loaded_size = desired

    model_status["stt"] = "ready"
    log.info("Whisper model loaded: %s", desired)
    return _model


def switch_model(size: str) -> None:
    """Switch to a different model size (clears current model)."""
    global _model, _loaded_size
    if size not in VALID_SIZES:
        raise ValueError(f"Invalid size {size!r}, must be one of {VALID_SIZES}")
    runtime_settings["stt_model_size"] = size
    _model = None
    _loaded_size = None
    model_status["stt"] = "not_loaded"
    model_status["stt_model"] = size


def download_model(size: str) -> None:
    """Pre-download a model by loading then discarding it."""
    if size not in VALID_SIZES:
        raise ValueError(f"Invalid size {size!r}, must be one of {VALID_SIZES}")

    from faster_whisper import WhisperModel

    model_status["stt"] = "loading"
    model_status["stt_model"] = size
    log.info("Pre-downloading faster-whisper model: %s ...", size)

    WhisperModel(size, device="cpu", compute_type="int8")

    # Restore status — only mark ready if this is the active size
    if runtime_settings.get("stt_model_size") == size:
        model_status["stt"] = "ready"
    else:
        model_status["stt"] = "not_loaded" if _model is None else "ready"
        model_status["stt_model"] = _loaded_size or size
    log.info("Model pre-downloaded: %s", size)


def get_status() -> dict:
    """Return current STT model status."""
    return {
        "status": model_status["stt"],
        "model": model_status["stt_model"],
        "loaded_size": _loaded_size,
    }


def transcribe(audio_bytes: bytes, sample_rate: int = 48000):
    """Transcribe PCM int16 audio bytes to text.

    Args:
        audio_bytes: Raw PCM int16 mono audio bytes.
        sample_rate: Sample rate of the audio (default 48kHz from WebRTC).

    Returns:
        Tuple of (text, no_speech_prob, avg_logprob).
        no_speech_prob: 0.0-1.0 (high = likely not speech).
        avg_logprob: negative float (closer to 0 = more confident).
    """
    if not audio_bytes:
        return "", 0.0, 0.0

    model = _get_model()

    # Convert int16 PCM to float32 normalized [-1.0, 1.0] — what faster-whisper expects
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    duration = len(samples) / sample_rate
    log.debug("Transcribing %.2fs of audio (%d samples @ %dHz)", duration, len(samples), sample_rate)

    # Resample to 16kHz — faster-whisper expects 16kHz input
    WHISPER_RATE = 16000
    if sample_rate != WHISPER_RATE:
        num_output = int(len(samples) * WHISPER_RATE / sample_rate)
        samples = resample(samples, num_output).astype(np.float32)
        log.debug("Resampled to %d samples @ %dHz", len(samples), WHISPER_RATE)

    segments, info = model.transcribe(samples, beam_size=5, language="en")

    # Collect all segment texts + confidence metrics
    text_parts = []
    worst_no_speech = 0.0
    avg_logprobs = []
    for segment in segments:
        text_parts.append(segment.text.strip())
        worst_no_speech = max(worst_no_speech, segment.no_speech_prob)
        avg_logprobs.append(segment.avg_logprob)

    result = " ".join(text_parts).strip()
    avg_logprob = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else 0.0
    log.info("Transcription: %r (no_speech=%.2f, avg_logprob=%.2f)",
             result[:100], worst_no_speech, avg_logprob)
    return result, worst_no_speech, avg_logprob
