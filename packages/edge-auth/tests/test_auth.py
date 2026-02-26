"""Tests for edge_auth — providers, middleware, composite."""

import pytest

from edge_auth.base import AuthProvider, AuthResult
from edge_auth.providers.bearer import BearerTokenProvider
from edge_auth.providers.none import NoAuthProvider
from edge_auth.composite import CompositeProvider


class FakeRequest:
    def __init__(self, headers=None, query_params=None):
        self.headers = headers or {}
        self.query_params = query_params or {}


class FakeWebSocket:
    def __init__(self, query_params=None):
        self.query_params = query_params or {}


class TestAuthResult:
    def test_defaults(self):
        r = AuthResult(authenticated=True)
        assert r.user_id is None
        assert r.provider == ""

    def test_with_fields(self):
        r = AuthResult(authenticated=True, user_id="123", provider="bearer")
        assert r.user_id == "123"


class TestNoAuthProvider:
    @pytest.mark.asyncio
    async def test_always_authenticates(self):
        auth = NoAuthProvider()
        result = await auth.authenticate(FakeRequest())
        assert result.authenticated is True
        assert result.provider == "none"

    @pytest.mark.asyncio
    async def test_ws_always_authenticates(self):
        auth = NoAuthProvider()
        result = await auth.authenticate_ws(FakeWebSocket())
        assert result.authenticated is True


class TestBearerTokenProvider:
    @pytest.mark.asyncio
    async def test_valid_token(self):
        auth = BearerTokenProvider(token="secret123")
        req = FakeRequest(headers={"authorization": "Bearer secret123"})
        result = await auth.authenticate(req)
        assert result.authenticated is True

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        auth = BearerTokenProvider(token="secret123")
        req = FakeRequest(headers={"authorization": "Bearer wrong"})
        result = await auth.authenticate(req)
        assert result.authenticated is False

    @pytest.mark.asyncio
    async def test_missing_header(self):
        auth = BearerTokenProvider(token="secret123")
        req = FakeRequest()
        result = await auth.authenticate(req)
        assert result.authenticated is False

    @pytest.mark.asyncio
    async def test_ws_via_query_param(self):
        auth = BearerTokenProvider(token="secret123")
        ws = FakeWebSocket(query_params={"token": "secret123"})
        result = await auth.authenticate_ws(ws)
        assert result.authenticated is True


class TestCompositeProvider:
    @pytest.mark.asyncio
    async def test_first_match_wins(self):
        auth = CompositeProvider([
            BearerTokenProvider(token="key1"),
            NoAuthProvider(),
        ])
        req = FakeRequest(headers={"authorization": "Bearer wrong"})
        result = await auth.authenticate(req)
        assert result.authenticated is True
        assert result.provider == "none"

    @pytest.mark.asyncio
    async def test_skips_none_providers(self):
        auth = CompositeProvider([None, NoAuthProvider()])
        result = await auth.authenticate(FakeRequest())
        assert result.authenticated is True

    @pytest.mark.asyncio
    async def test_all_fail(self):
        auth = CompositeProvider([
            BearerTokenProvider(token="secret"),
        ])
        req = FakeRequest()
        result = await auth.authenticate(req)
        assert result.authenticated is False
