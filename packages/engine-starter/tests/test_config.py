"""Tests for engine_starter.config — EngineConfig runtime store."""

import pytest

from engine_starter.config import EngineConfig


class TestDefaults:
    def test_has_all_expected_keys(self):
        cfg = EngineConfig()
        snap = cfg.snapshot()
        expected = {
            "tts_engine", "tts_voice",
            "vad_energy_threshold", "vad_speech_confirm_frames", "vad_silence_gap",
            "barge_in_enabled", "barge_in_energy_threshold", "barge_in_confirm_frames",
        }
        assert set(snap.keys()) == expected

    def test_default_values(self):
        cfg = EngineConfig()
        assert cfg["tts_engine"] == "piper"
        assert cfg["barge_in_enabled"] is True


class TestGetSet:
    def test_set_and_get(self):
        cfg = EngineConfig()
        cfg["vad_energy_threshold"] = 800
        assert cfg["vad_energy_threshold"] == 800

    def test_get_with_default(self):
        cfg = EngineConfig()
        assert cfg.get("vad_energy_threshold") == cfg["vad_energy_threshold"]
        assert cfg.get("nonexistent", 42) == 42

    def test_unknown_key_raises(self):
        cfg = EngineConfig()
        with pytest.raises(KeyError):
            cfg["not_a_real_key"] = 123

    def test_unknown_key_get_raises(self):
        cfg = EngineConfig()
        with pytest.raises(KeyError):
            _ = cfg["not_a_real_key"]


class TestUpdate:
    def test_update_returns_full_config(self):
        cfg = EngineConfig()
        result = cfg.update({"vad_energy_threshold": 900, "tts_voice": "alba"})
        assert result["vad_energy_threshold"] == 900
        assert result["tts_voice"] == "alba"
        # also returns non-updated keys
        assert "barge_in_enabled" in result

    def test_update_unknown_key_raises(self):
        cfg = EngineConfig()
        with pytest.raises(KeyError):
            cfg.update({"bogus_key": True})


class TestSnapshot:
    def test_snapshot_is_a_copy(self):
        cfg = EngineConfig()
        snap = cfg.snapshot()
        snap["tts_engine"] = "modified"
        assert cfg["tts_engine"] != "modified"
