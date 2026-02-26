"""GoogleJWTProvider — validates Google OAuth2 ID tokens."""

from __future__ import annotations

import logging
import os
from typing import Optional, Set

from edge_auth.base import AuthProvider, AuthResult

log = logging.getLogger("edge_auth.google")


class GoogleJWTProvider(AuthProvider):
    """Validate Google Sign-In JWT tokens.

    Requires: pip install voice-frontend-edge-auth[google]
    """

    def __init__(self, client_id: str = "", allowed_emails: Optional[Set[str]] = None):
        self._client_id = client_id or os.getenv("GOOGLE_CLIENT_ID", "")
        self._allowed_emails = allowed_emails

    async def authenticate(self, request) -> AuthResult:
        auth_header = getattr(request, 'headers', {}).get("authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
        if not token:
            return AuthResult(authenticated=False, provider="google")
        return self._verify(token)

    async def authenticate_ws(self, websocket) -> AuthResult:
        token = getattr(websocket, 'query_params', {}).get("google_jwt", "")
        if not token:
            return AuthResult(authenticated=False, provider="google")
        return self._verify(token)

    def _verify(self, token: str) -> AuthResult:
        try:
            from google.auth.transport import requests as google_requests
            from google.oauth2 import id_token

            idinfo = id_token.verify_oauth2_token(
                token, google_requests.Request(), self._client_id
            )

            if idinfo.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
                return AuthResult(authenticated=False, provider="google")

            if not idinfo.get("email_verified"):
                return AuthResult(authenticated=False, provider="google")

            email = idinfo.get("email", "").lower()
            if self._allowed_emails and email not in self._allowed_emails:
                return AuthResult(authenticated=False, provider="google")

            return AuthResult(
                authenticated=True,
                user_id=idinfo.get("sub"),
                user_email=email,
                user_name=idinfo.get("name"),
                provider="google",
            )
        except Exception as e:
            log.warning("Google auth failed: %s", e)
            return AuthResult(authenticated=False, provider="google")
