"""FastAPI middleware and dependency for auth."""

from __future__ import annotations

from typing import List, Optional

from edge_auth.base import AuthProvider, AuthResult


def auth_middleware(provider: AuthProvider, exclude: Optional[List[str]] = None):
    """Create Starlette middleware that authenticates all requests."""
    exclude = exclude or []

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class _AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            for prefix in exclude:
                if request.url.path.startswith(prefix):
                    return await call_next(request)
            result = await provider.authenticate(request)
            if not result.authenticated:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            request.state.auth = result
            return await call_next(request)

    return _AuthMiddleware


def auth_dependency(provider: AuthProvider):
    """Create a FastAPI dependency that authenticates the request."""
    from fastapi import HTTPException, Request

    async def _dependency(request: Request) -> AuthResult:
        result = await provider.authenticate(request)
        if not result.authenticated:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return result

    return _dependency
