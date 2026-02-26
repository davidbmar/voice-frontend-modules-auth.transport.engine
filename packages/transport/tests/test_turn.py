"""Tests for transport.turn — TURN credential providers."""

import pytest

from transport.turn import TURNProvider, TwilioTURN, StaticICE


class TestStaticICE:
    @pytest.mark.asyncio
    async def test_returns_configured_servers(self):
        servers = [{"urls": "stun:stun.example.com:3478"}]
        provider = StaticICE(servers=servers)
        result = await provider.fetch_ice_servers()
        assert result == servers

    @pytest.mark.asyncio
    async def test_empty_list(self):
        provider = StaticICE(servers=[])
        result = await provider.fetch_ice_servers()
        assert result == []


class TestTwilioTURN:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        provider = TwilioTURN()
        result = await provider.fetch_ice_servers()
        assert result == []

    def test_is_turn_provider(self):
        assert issubclass(TwilioTURN, TURNProvider)
        assert issubclass(StaticICE, TURNProvider)
