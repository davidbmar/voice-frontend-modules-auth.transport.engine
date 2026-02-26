"""Tests for engine_starter.interfaces — provider ABCs."""

import pytest

from engine_starter.interfaces import (
    STTProvider,
    TTSProvider,
    LLMProvider,
    TranscriptionResult,
    AudioChunk,
)


class TestTranscriptionResult:
    def test_defaults(self):
        r = TranscriptionResult(text="hello")
        assert r.text == "hello"
        assert r.no_speech_probability == 0.0
        assert r.language == ""

    def test_with_all_fields(self):
        r = TranscriptionResult(text="hello", no_speech_probability=0.1, language="en")
        assert r.no_speech_probability == 0.1


class TestAudioChunk:
    def test_fields(self):
        chunk = AudioChunk(samples=b"\x00\x00", sample_rate=48000, channels=1)
        assert chunk.sample_rate == 48000
        assert chunk.channels == 1


class TestABCsCannotBeInstantiated:
    def test_stt_is_abstract(self):
        with pytest.raises(TypeError):
            STTProvider()

    def test_tts_is_abstract(self):
        with pytest.raises(TypeError):
            TTSProvider()

    def test_llm_is_abstract(self):
        with pytest.raises(TypeError):
            LLMProvider()
