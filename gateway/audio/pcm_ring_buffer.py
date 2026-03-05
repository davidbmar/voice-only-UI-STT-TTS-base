"""Thread-safe ring buffer for PCM audio data.

Decouples audio producers (TTS engines running in threads) from
the WebRTC consumer (which pulls every 20ms in the async loop).
Not used for sine waves (which are computed inline), but ready
for real TTS integration.
"""

import threading


class PCMRingBuffer:
    """Fixed-size byte ring buffer with thread-safe read/write.

    - write() appends data, silently discarding oldest bytes on overflow
    - read(n) returns exactly n bytes, zero-padding if not enough data
    """

    def __init__(self, capacity: int = 48000 * 2 * 2):
        # Default: ~1 second of 48kHz mono 16-bit audio
        self._buf = bytearray(capacity)
        self._capacity = capacity
        self._write_pos = 0
        self._read_pos = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        """Bytes available for reading."""
        with self._lock:
            return self._size

    def write(self, data: bytes) -> int:
        """Write data into the buffer. Returns bytes actually written.

        If buffer is full, oldest data is overwritten (lossy).
        """
        with self._lock:
            n = len(data)
            for i in range(n):
                self._buf[self._write_pos] = data[i]
                self._write_pos = (self._write_pos + 1) % self._capacity
                if self._size == self._capacity:
                    # Overwriting unread data — advance read pointer
                    self._read_pos = (self._read_pos + 1) % self._capacity
                else:
                    self._size += 1
            return n

    def read(self, n: int) -> bytes:
        """Read up to n bytes. Zero-pads if fewer bytes are available."""
        with self._lock:
            available = min(n, self._size)
            result = bytearray(n)
            for i in range(available):
                result[i] = self._buf[self._read_pos]
                self._read_pos = (self._read_pos + 1) % self._capacity
            self._size -= available
            # Remaining bytes in result are already 0 (silence)
            return bytes(result)

    def clear(self):
        """Discard all buffered data."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._size = 0
