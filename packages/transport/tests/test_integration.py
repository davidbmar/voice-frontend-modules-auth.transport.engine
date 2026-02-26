# packages/transport/tests/test_integration.py
"""Integration tests — verify all public APIs import correctly."""

import pytest
import numpy as np

from transport.audio import AudioQueue, resample


class TestFullPipelineImports:
    """Verify all public APIs can be imported."""

    def test_transport_imports(self):
        from transport.turn import TURNProvider, TwilioTURN, StaticICE
        from transport.signaling import SignalingServer
        from transport.session import WebRTCSession
        from transport.tunnel import CloudflareTunnel
        from transport.audio import AudioQueue, resample

    def test_edge_auth_imports(self):
        from edge_auth import AuthProvider, AuthResult, auth_middleware, auth_dependency, CompositeProvider
        from edge_auth.providers import CloudflareAccessProvider, GoogleJWTProvider, BearerTokenProvider, NoAuthProvider

    def test_engine_starter_imports(self):
        from engine_starter import STTProvider, TTSProvider, LLMProvider, TranscriptionResult
        from engine_starter.interfaces import AudioChunk


class TestResampleRoundTrip:
    def test_48k_to_16k_and_back(self):
        original = np.random.randint(-1000, 1000, size=960, dtype=np.int16).tobytes()
        down = resample(original, from_rate=48000, to_rate=16000)
        up = resample(down, from_rate=16000, to_rate=48000)
        assert len(np.frombuffer(up, dtype=np.int16)) == 960
