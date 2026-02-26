"""Tests for transport.signaling — WebSocket signaling server."""

import json
import pytest

from transport.signaling import SignalingServer
from transport.turn import StaticICE


class FakeWebSocket:
    """Minimal mock for FastAPI WebSocket."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self._sent: list[dict] = []
        self._accepted = False

    async def accept(self):
        self._accepted = True

    async def receive_text(self) -> str:
        if not self._messages:
            raise Exception("disconnected")
        return self._messages.pop(0)

    async def send_json(self, data: dict):
        self._sent.append(data)


class TestSignalingServer:
    @pytest.mark.asyncio
    async def test_hello_returns_ice_servers(self):
        ice = [{"urls": "stun:stun.example.com:3478"}]
        server = SignalingServer(
            turn_provider=StaticICE(servers=ice),
            on_session=lambda session: None,
        )
        ws = FakeWebSocket([json.dumps({"type": "hello"})])
        await server.handle(ws)

        assert ws._accepted
        assert len(ws._sent) >= 1
        assert ws._sent[0]["type"] == "hello_ack"
        assert ws._sent[0]["ice_servers"] == ice

    @pytest.mark.asyncio
    async def test_ping_returns_pong(self):
        server = SignalingServer(
            turn_provider=StaticICE(),
            on_session=lambda session: None,
        )
        ws = FakeWebSocket([
            json.dumps({"type": "hello"}),
            json.dumps({"type": "ping"}),
        ])
        await server.handle(ws)

        types = [m["type"] for m in ws._sent]
        assert "pong" in types

    @pytest.mark.asyncio
    async def test_unknown_type_returns_error(self):
        server = SignalingServer(
            turn_provider=StaticICE(),
            on_session=lambda session: None,
        )
        ws = FakeWebSocket([
            json.dumps({"type": "hello"}),
            json.dumps({"type": "unknown_thing"}),
        ])
        await server.handle(ws)

        error_msgs = [m for m in ws._sent if m["type"] == "error"]
        assert len(error_msgs) == 1

    @pytest.mark.asyncio
    async def test_hangup_ends_loop(self):
        server = SignalingServer(
            turn_provider=StaticICE(),
            on_session=lambda session: None,
        )
        ws = FakeWebSocket([
            json.dumps({"type": "hello"}),
            json.dumps({"type": "hangup"}),
        ])
        await server.handle(ws)
        # Should complete without error, only hello_ack sent
        assert ws._sent[0]["type"] == "hello_ack"
        assert len(ws._sent) == 1
