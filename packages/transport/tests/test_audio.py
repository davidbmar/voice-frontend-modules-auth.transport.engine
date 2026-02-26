"""Tests for transport.audio — AudioQueue and resample utilities."""

import numpy as np
import pytest

from transport.audio import AudioQueue, resample


class TestAudioQueue:
    def test_read_empty_returns_silence(self):
        q = AudioQueue()
        data = q.read(100)
        assert len(data) == 100
        assert data == b"\x00" * 100

    def test_enqueue_and_read_exact(self):
        q = AudioQueue()
        q.enqueue(b"\x01\x02\x03\x04")
        assert q.read(4) == b"\x01\x02\x03\x04"

    def test_read_pads_silence_when_not_enough_data(self):
        q = AudioQueue()
        q.enqueue(b"\xff\xff")
        data = q.read(4)
        assert data == b"\xff\xff\x00\x00"

    def test_read_spans_multiple_chunks(self):
        q = AudioQueue()
        q.enqueue(b"\x01\x02")
        q.enqueue(b"\x03\x04")
        assert q.read(4) == b"\x01\x02\x03\x04"

    def test_clear_discards_all(self):
        q = AudioQueue()
        q.enqueue(b"\x01\x02\x03\x04")
        q.clear()
        assert q.available == 0
        assert q.read(4) == b"\x00\x00\x00\x00"

    def test_available_property(self):
        q = AudioQueue()
        assert q.available == 0
        q.enqueue(b"\x01\x02")
        assert q.available == 2
        q.read(1)
        assert q.available == 1

    def test_enqueue_empty_is_noop(self):
        q = AudioQueue()
        q.enqueue(b"")
        assert q.available == 0


class TestResample:
    def test_downsample_48k_to_16k(self):
        # 960 samples at 48kHz = 20ms -> should become 320 samples at 16kHz
        samples_48k = np.zeros(960, dtype=np.int16)
        samples_48k[0] = 1000
        result = resample(samples_48k.tobytes(), from_rate=48000, to_rate=16000)
        result_arr = np.frombuffer(result, dtype=np.int16)
        assert len(result_arr) == 320

    def test_upsample_16k_to_48k(self):
        samples_16k = np.zeros(320, dtype=np.int16)
        result = resample(samples_16k.tobytes(), from_rate=16000, to_rate=48000)
        result_arr = np.frombuffer(result, dtype=np.int16)
        assert len(result_arr) == 960

    def test_same_rate_is_passthrough(self):
        pcm = np.array([100, 200, 300], dtype=np.int16).tobytes()
        result = resample(pcm, from_rate=48000, to_rate=48000)
        assert result == pcm
