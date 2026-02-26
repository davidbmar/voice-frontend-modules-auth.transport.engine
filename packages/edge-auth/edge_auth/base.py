"""Auth provider base class and result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class AuthResult:
    """Result of an authentication attempt."""
    authenticated: bool
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    provider: str = ""


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    async def authenticate(self, request) -> AuthResult:
        """Validate an HTTP request."""
        ...

    async def authenticate_ws(self, websocket) -> AuthResult:
        """Validate a WebSocket. Default: delegates to authenticate."""
        return await self.authenticate(websocket)
