"""Audio generation adapter — sine waves now, real TTS later."""

import math
import struct
from typing import List

from .types import AudioChunk, VoiceInfo

SAMPLE_RATE = 48000
FRAME_SAMPLES = 960  # 20ms at 48kHz

# Each "voice" is a different sine wave frequency
VOICES: List[VoiceInfo] = [
    VoiceInfo(id="sine-220", name="Low Tone (220 Hz)", description="A low A3 sine wave"),
    VoiceInfo(id="sine-440", name="Mid Tone (440 Hz)", description="Concert pitch A4 sine wave"),
    VoiceInfo(id="sine-880", name="High Tone (880 Hz)", description="A high A5 sine wave"),
]

VOICE_FREQ = {
    "sine-220": 220.0,
    "sine-440": 440.0,
    "sine-880": 880.0,
}


def list_voices() -> List[VoiceInfo]:
    """Return available voices."""
    return VOICES


class SineWaveGenerator:
    """Generates continuous sine wave PCM audio chunks.

    Maintains phase continuity across calls to next_chunk() so there
    are no clicks or pops at frame boundaries.
    """

    def __init__(self, voice_id: str, amplitude: float = 0.3):
        if voice_id not in VOICE_FREQ:
            raise ValueError(f"Unknown voice: {voice_id}")
        self.frequency = VOICE_FREQ[voice_id]
        self.amplitude = amplitude
        self.phase = 0.0

    def next_chunk(self) -> AudioChunk:
        """Generate the next 20ms frame of audio."""
        phase_inc = 2.0 * math.pi * self.frequency / SAMPLE_RATE
        samples = []
        for _ in range(FRAME_SAMPLES):
            value = self.amplitude * math.sin(self.phase)
            # Clamp to int16 range and pack as little-endian signed 16-bit
            samples.append(int(max(-32768, min(32767, value * 32767))))
            self.phase += phase_inc

        # Keep phase in [0, 2pi) to avoid floating point drift
        self.phase %= 2.0 * math.pi

        pcm = struct.pack(f"<{FRAME_SAMPLES}h", *samples)
        return AudioChunk(samples=pcm, sample_rate=SAMPLE_RATE, channels=1)


def create_generator(voice_id: str) -> SineWaveGenerator:
    """Factory to create an audio generator for the given voice."""
    return SineWaveGenerator(voice_id)
