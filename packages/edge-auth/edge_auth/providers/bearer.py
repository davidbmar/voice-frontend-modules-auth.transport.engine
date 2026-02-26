"""BearerTokenProvider — simple API key in Authorization header."""

from edge_auth.base import AuthProvider, AuthResult


class BearerTokenProvider(AuthProvider):
    def __init__(self, token: str):
        self._token = token

    async def authenticate(self, request) -> AuthResult:
        auth_header = getattr(request, 'headers', {}).get("authorization", "")
        if auth_header.startswith("Bearer ") and auth_header[7:] == self._token:
            return AuthResult(authenticated=True, provider="bearer")
        return AuthResult(authenticated=False, provider="bearer")

    async def authenticate_ws(self, websocket) -> AuthResult:
        token = getattr(websocket, 'query_params', {}).get("token", "")
        if token == self._token:
            return AuthResult(authenticated=True, provider="bearer")
        return AuthResult(authenticated=False, provider="bearer")
