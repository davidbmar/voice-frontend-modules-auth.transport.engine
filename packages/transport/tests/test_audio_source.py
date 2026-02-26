"""Tests for transport.audio_source — WebRTC audio track."""

import numpy as np
import pytest
from dataclasses import dataclass

from transport.audio_source import WebRTCAudioSource, FRAME_SAMPLES, SAMPLE_RATE


@dataclass
class AudioChunk:
    samples: bytes
    sample_rate: int
    channels: int


class FakeGenerator:
    """Fake audio generator that returns constant PCM data."""
    def __init__(self, value: int = 1):
        self._value = value

    def next_chunk(self):
        pcm = np.full(FRAME_SAMPLES, self._value, dtype=np.int16).tobytes()
        return AudioChunk(samples=pcm, sample_rate=SAMPLE_RATE, channels=1)


class TestWebRTCAudioSource:
    @pytest.mark.asyncio
    async def test_recv_returns_silence_without_generator(self):
        source = WebRTCAudioSource()
        frame = await source.recv()
        assert frame.sample_rate == SAMPLE_RATE
        arr = frame.to_ndarray().flatten()
        assert len(arr) == FRAME_SAMPLES
        assert np.all(arr == 0)

    @pytest.mark.asyncio
    async def test_recv_returns_generator_audio(self):
        source = WebRTCAudioSource()
        source.set_generator(FakeGenerator(value=42))
        frame = await source.recv()
        arr = frame.to_ndarray().flatten()
        assert np.all(arr == 42)

    @pytest.mark.asyncio
    async def test_clear_generator_reverts_to_silence(self):
        source = WebRTCAudioSource()
        source.set_generator(FakeGenerator(value=42))
        await source.recv()  # generator audio
        source.clear_generator()
        frame = await source.recv()
        arr = frame.to_ndarray().flatten()
        assert np.all(arr == 0)

    @pytest.mark.asyncio
    async def test_frame_format(self):
        source = WebRTCAudioSource()
        frame = await source.recv()
        assert frame.format.name == "s16"
        assert frame.sample_rate == 48000
