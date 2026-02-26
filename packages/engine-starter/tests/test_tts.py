"""Tests for engine_starter.tts — StarterTTS."""

import pytest

from engine_starter.interfaces import TTSProvider
from engine_starter.tts import StarterTTS, VOICE_CATALOG


class TestStarterTTS:
    def test_is_tts_provider(self):
        assert issubclass(StarterTTS, TTSProvider)

    def test_list_voices_returns_strings(self):
        tts = StarterTTS()
        voices = tts.list_voices()
        assert len(voices) > 0
        assert all(isinstance(v, str) for v in voices)

    def test_synthesize_empty_returns_empty(self):
        tts = StarterTTS()
        result = tts.synthesize("")
        assert result == b""


class TestVoiceCatalog:
    def test_has_default_voice(self):
        ids = [v["id"] for v in VOICE_CATALOG]
        assert "en_US-lessac-medium" in ids

    def test_all_entries_have_required_fields(self):
        for v in VOICE_CATALOG:
            assert "id" in v
            assert "name" in v
            assert "lang" in v
