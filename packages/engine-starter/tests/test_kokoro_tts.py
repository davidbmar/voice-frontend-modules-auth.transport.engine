"""Tests for engine_starter.kokoro_tts — KokoroTTS provider."""

import importlib

import pytest

from engine_starter.interfaces import TTSProvider
from engine_starter.kokoro_tts import KokoroTTS, VOICE_GROUPS, LANG_MAP, MODEL_DIR

_has_kokoro = importlib.util.find_spec("kokoro_onnx") is not None
_has_models = (MODEL_DIR / "kokoro-v1.0.onnx").exists() and (MODEL_DIR / "voices.bin").exists()


class TestKokoroTTS:
    def test_is_tts_provider(self):
        assert issubclass(KokoroTTS, TTSProvider)

    def test_synthesize_empty_returns_empty(self):
        tts = KokoroTTS()
        assert tts.synthesize("") == b""
        assert tts.synthesize("   ") == b""

    def test_voice_groups_dict(self):
        assert isinstance(VOICE_GROUPS, dict)
        assert len(VOICE_GROUPS) > 0
        for prefix, label in VOICE_GROUPS.items():
            assert isinstance(prefix, str)
            assert isinstance(label, str)

    def test_lang_map(self):
        assert "a" in LANG_MAP  # US English
        assert "b" in LANG_MAP  # British English
        assert isinstance(LANG_MAP["a"], str)

    @pytest.mark.skipif(not _has_kokoro or not _has_models, reason="kokoro-onnx not installed or model files missing")
    def test_list_voices(self):
        tts = KokoroTTS()
        voices = tts.list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert all(isinstance(v, str) for v in voices)
        assert voices == sorted(voices)
