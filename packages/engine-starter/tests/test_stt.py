"""Tests for engine_starter.stt — StarterSTT."""

import importlib

import numpy as np
import pytest

from engine_starter.interfaces import STTProvider, TranscriptionResult
from engine_starter.stt import StarterSTT

_has_faster_whisper = importlib.util.find_spec("faster_whisper") is not None


class TestStarterSTT:
    def test_is_stt_provider(self):
        assert issubclass(StarterSTT, STTProvider)

    def test_transcribe_empty_returns_empty(self):
        stt = StarterSTT()
        result = stt.transcribe(b"", sample_rate=16000)
        assert isinstance(result, TranscriptionResult)
        assert result.text == ""

    @pytest.mark.skipif(
        not _has_faster_whisper,
        reason="faster-whisper not installed",
    )
    def test_transcribe_returns_transcription_result(self):
        """Verify return type even with silence."""
        silence = np.zeros(16000, dtype=np.int16).tobytes()
        stt = StarterSTT()
        result = stt.transcribe(silence, sample_rate=16000)
        assert isinstance(result, TranscriptionResult)
