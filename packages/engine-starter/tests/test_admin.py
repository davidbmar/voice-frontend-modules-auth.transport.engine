"""Tests for engine_starter.admin — Admin API router."""

import struct

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engine_starter.config import EngineConfig
from engine_starter.interfaces import TTSProvider


class MockTTS(TTSProvider):
    """Minimal TTS provider for testing."""

    def synthesize(self, text: str, voice: str = "") -> bytes:
        if not text:
            return b""
        # Return fake 48kHz int16 PCM (480 samples = 10ms)
        return b"\x00\x01" * 480

    def list_voices(self) -> list[str]:
        return ["voice_a", "voice_b"]


@pytest.fixture
def app():
    from engine_starter.admin import create_admin_router

    config = EngineConfig()
    tts_providers = {"mock": MockTTS()}
    router = create_admin_router(config=config, tts_providers=tts_providers)

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestGetConfig:
    def test_returns_all_keys(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "tts_engine" in data
        assert "barge_in_enabled" in data

    def test_returns_default_values(self, client):
        resp = client.get("/api/config")
        data = resp.json()
        assert data["tts_engine"] == "piper"
        assert data["barge_in_enabled"] is True


class TestPostConfig:
    def test_partial_update(self, client):
        resp = client.post("/api/config", json={"vad_energy_threshold": 900})
        assert resp.status_code == 200
        data = resp.json()
        assert data["vad_energy_threshold"] == 900
        # Other keys unchanged
        assert data["tts_engine"] == "piper"

    def test_unknown_key_returns_400(self, client):
        resp = client.post("/api/config", json={"bogus": 123})
        assert resp.status_code == 400

    def test_config_persists(self, client):
        client.post("/api/config", json={"tts_voice": "alba"})
        resp = client.get("/api/config")
        assert resp.json()["tts_voice"] == "alba"


class TestListVoices:
    def test_returns_provider_voices(self, client):
        resp = client.get("/api/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert "mock" in data
        assert data["mock"] == ["voice_a", "voice_b"]


class TestTTSPreview:
    def test_returns_wav(self, client):
        resp = client.post(
            "/api/tts/preview",
            json={"text": "hello", "engine": "mock"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        # Verify WAV header starts with RIFF
        assert resp.content[:4] == b"RIFF"

    def test_missing_engine_returns_400(self, client):
        resp = client.post("/api/tts/preview", json={"text": "hello"})
        assert resp.status_code == 400

    def test_unknown_engine_returns_404(self, client):
        resp = client.post(
            "/api/tts/preview",
            json={"text": "hello", "engine": "nonexistent"},
        )
        assert resp.status_code == 404


class TestAdminPage:
    def test_serves_html(self, client):
        resp = client.get("/admin")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
