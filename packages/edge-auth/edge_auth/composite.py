"""CompositeProvider — try multiple auth providers in order."""

from edge_auth.base import AuthProvider, AuthResult


class CompositeProvider(AuthProvider):
    """Try multiple providers in order. First successful auth wins."""

    def __init__(self, providers: list):
        self._providers = [p for p in providers if p is not None]

    async def authenticate(self, request) -> AuthResult:
        for provider in self._providers:
            result = await provider.authenticate(request)
            if result.authenticated:
                return result
        return AuthResult(authenticated=False)

    async def authenticate_ws(self, websocket) -> AuthResult:
        for provider in self._providers:
            result = await provider.authenticate_ws(websocket)
            if result.authenticated:
                return result
        return AuthResult(authenticated=False)
