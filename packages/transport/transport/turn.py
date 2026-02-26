"""TURN/STUN credential providers for WebRTC ICE negotiation.

TwilioTURN fetches ephemeral credentials from Twilio's Network Traversal Service.
StaticICE returns a fixed list of servers (for dev/testing or self-hosted coturn).
"""

import logging
import os
from abc import ABC, abstractmethod

import aiohttp

log = logging.getLogger("transport.turn")


class TURNProvider(ABC):
    """Abstract base class for TURN/STUN credential providers."""

    @abstractmethod
    async def fetch_ice_servers(self) -> list[dict]:
        """Return ICE server dicts in WebRTC format."""
        ...


class TwilioTURN(TURNProvider):
    """Fetch ephemeral TURN credentials from Twilio NTS.

    Reads TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN from environment.
    Returns [] if not configured or request fails.
    """

    def __init__(self, account_sid: str = "", auth_token: str = ""):
        self._account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")
        self._auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN", "")

    async def fetch_ice_servers(self) -> list[dict]:
        if not self._account_sid or not self._auth_token:
            log.warning("TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not set — no TURN servers")
            return []

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self._account_sid}/Tokens.json"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    auth=aiohttp.BasicAuth(self._account_sid, self._auth_token),
                ) as resp:
                    if resp.status != 201:
                        body = await resp.text()
                        log.error("Twilio token request failed (%d): %s", resp.status, body)
                        return []
                    data = await resp.json()

            ice_servers = data.get("ice_servers", [])
            log.info("Got %d ICE servers from Twilio (TTL: %ss)",
                     len(ice_servers), data.get("ttl", "?"))
            return ice_servers

        except Exception as e:
            log.error("Failed to fetch Twilio TURN credentials: %s", e)
            return []


class StaticICE(TURNProvider):
    """Return a fixed list of STUN/TURN servers.

    Useful for development (STUN-only) or self-hosted coturn deployments.
    """

    def __init__(self, servers: list[dict] | None = None):
        self._servers = servers if servers is not None else [
            {"urls": "stun:stun.l.google.com:19302"},
        ]

    async def fetch_ice_servers(self) -> list[dict]:
        return self._servers
