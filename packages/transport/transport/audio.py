"""Audio utilities for WebRTC voice transport.

AudioQueue: Thread-safe FIFO for TTS output (never drops audio).
resample: Convert PCM between sample rates (e.g., 48kHz WebRTC <-> 16kHz STT).
"""

import threading
from collections import deque

import numpy as np


class AudioQueue:
    """Unbounded FIFO of PCM byte blobs, read out in fixed-size chunks.

    Producers call enqueue() with variable-length PCM blobs (one per TTS sentence).
    The consumer calls read(n) every 20ms to get exactly n bytes, zero-padded
    if not enough data is available yet.
    """

    def __init__(self):
        self._chunks: deque[bytes] = deque()
        self._current = b""
        self._offset = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        """Total bytes available for reading."""
        with self._lock:
            total = len(self._current) - self._offset
            for chunk in self._chunks:
                total += len(chunk)
            return total

    def enqueue(self, data: bytes):
        """Append a PCM blob to the queue (thread-safe)."""
        if not data:
            return
        with self._lock:
            self._chunks.append(data)

    def read(self, n: int) -> bytes:
        """Read exactly n bytes. Zero-pads if not enough data available."""
        with self._lock:
            result = bytearray(n)
            written = 0
            while written < n:
                if self._offset >= len(self._current):
                    if not self._chunks:
                        break
                    self._current = self._chunks.popleft()
                    self._offset = 0
                remaining_in_chunk = len(self._current) - self._offset
                to_copy = min(remaining_in_chunk, n - written)
                result[written:written + to_copy] = self._current[self._offset:self._offset + to_copy]
                self._offset += to_copy
                written += to_copy
            return bytes(result)

    @property
    def is_playing(self) -> bool:
        """True if there is audio queued for playback."""
        return self.available > 0

    def clear(self):
        """Discard all queued audio."""
        with self._lock:
            self._chunks.clear()
            self._current = b""
            self._offset = 0


def resample(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample int16 PCM between sample rates.

    Args:
        pcm_bytes: Raw PCM int16 mono audio bytes.
        from_rate: Source sample rate (e.g., 48000).
        to_rate: Target sample rate (e.g., 16000).

    Returns:
        Resampled PCM int16 bytes.
    """
    if from_rate == to_rate:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
    num_output = int(len(samples) * to_rate / from_rate)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, num_output),
        np.arange(len(samples)),
        samples,
    )
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()
