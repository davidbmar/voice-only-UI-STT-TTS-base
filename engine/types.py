"""Shared data types for the audio engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceInfo:
    """Describes an available voice."""
    id: str
    name: str
    description: str


@dataclass
class AudioChunk:
    """A chunk of raw PCM audio data."""
    samples: bytes          # 16-bit signed LE PCM
    sample_rate: int = 48000
    channels: int = 1
