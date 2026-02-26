"""Tests for barge-in detection in transport session."""

import inspect

import numpy as np
import pytest

from transport.audio import AudioQueue


class TestComputeRMS:
    """Test WebRTCSession._compute_rms static method."""

    def test_silence_is_zero(self):
        from transport.session import WebRTCSession
        silence = np.zeros(960, dtype=np.int16).tobytes()
        assert WebRTCSession._compute_rms(silence) == 0.0

    def test_loud_signal(self):
        from transport.session import WebRTCSession
        loud = (np.ones(960, dtype=np.int16) * 10000).tobytes()
        rms = WebRTCSession._compute_rms(loud)
        assert rms > 5000

    def test_empty_bytes(self):
        from transport.session import WebRTCSession
        assert WebRTCSession._compute_rms(b"") == 0.0


class TestAudioQueueIsPlaying:
    def test_empty_queue_not_playing(self):
        q = AudioQueue()
        assert q.is_playing is False

    def test_queue_with_data_is_playing(self):
        q = AudioQueue()
        q.enqueue(b"\x00\x01" * 100)
        assert q.is_playing is True

    def test_after_drain_not_playing(self):
        q = AudioQueue()
        q.enqueue(b"\x00\x01" * 10)
        q.read(20)  # drain all
        assert q.is_playing is False


class TestBargeInAttributes:
    def test_compute_rms_is_static(self):
        """Verify _compute_rms exists and is callable."""
        from transport.session import WebRTCSession
        assert hasattr(WebRTCSession, '_compute_rms')
        assert callable(WebRTCSession._compute_rms)

    def test_speak_with_barge_in_signature(self):
        """Verify speak_with_barge_in has the expected parameters."""
        from transport.session import WebRTCSession
        sig = inspect.signature(WebRTCSession.speak_with_barge_in)
        params = set(sig.parameters.keys()) - {"self"}
        assert "text" in params
        assert "tts" in params
        assert "voice" in params
        assert "stt" in params
