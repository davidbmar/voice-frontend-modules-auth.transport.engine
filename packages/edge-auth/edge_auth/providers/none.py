"""NoAuthProvider — allows everything. For local development only."""

from edge_auth.base import AuthProvider, AuthResult


class NoAuthProvider(AuthProvider):
    async def authenticate(self, request) -> AuthResult:
        return AuthResult(authenticated=True, provider="none")

    async def authenticate_ws(self, websocket) -> AuthResult:
        return AuthResult(authenticated=True, provider="none")
