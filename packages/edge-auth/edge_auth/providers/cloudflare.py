"""CloudflareAccessProvider — validates CF-Access-JWT-Assertion header."""

from __future__ import annotations

import logging

from edge_auth.base import AuthProvider, AuthResult

log = logging.getLogger("edge_auth.cloudflare")


class CloudflareAccessProvider(AuthProvider):
    """Validate Cloudflare Access JWT tokens.

    Requires: pip install voice-frontend-edge-auth[cloudflare]
    """

    def __init__(self, team_domain: str, audience: str = ""):
        self._team_domain = team_domain
        self._audience = audience
        self._certs_url = f"https://{team_domain}/cdn-cgi/access/certs"

    async def authenticate(self, request) -> AuthResult:
        token = getattr(request, 'headers', {}).get("cf-access-jwt-assertion", "")
        if not token:
            return AuthResult(authenticated=False, provider="cloudflare")

        try:
            import jwt
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(self._certs_url)
                resp.raise_for_status()
                jwks = resp.json()

            public_keys = jwt.PyJWKSet.from_dict(jwks)
            header = jwt.get_unverified_header(token)
            key = public_keys[header["kid"]]

            payload = jwt.decode(
                token, key=key.key, algorithms=["RS256"],
                audience=self._audience or None,
            )

            return AuthResult(
                authenticated=True,
                user_email=payload.get("email"),
                provider="cloudflare",
            )
        except Exception as e:
            log.warning("CF Access auth failed: %s", e)
            return AuthResult(authenticated=False, provider="cloudflare")
