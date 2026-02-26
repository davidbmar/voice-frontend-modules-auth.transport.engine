"""Admin API router — reusable FastAPI router for engine configuration.

Provides endpoints for live-tuning engine settings, listing voices,
and previewing TTS output. Mount it in any FastAPI app::

    from engine_starter.admin import create_admin_router
    from engine_starter.config import EngineConfig

    config = EngineConfig()
    app.include_router(create_admin_router(config=config, tts_providers={"piper": tts}))

The admin panel HTML is served from ``engine_starter/static/admin.html``.
"""

import struct
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

from engine_starter.config import EngineConfig

_STATIC_DIR = Path(__file__).parent / "static"

# WAV header constants (48kHz mono int16)
_WAV_SAMPLE_RATE = 48000
_WAV_CHANNELS = 1
_WAV_BITS_PER_SAMPLE = 16


def _build_wav(pcm: bytes) -> bytes:
    """Wrap raw int16 PCM in a WAV header (48kHz mono)."""
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,              # file size - 8
        b"WAVE",
        b"fmt ",
        16,                          # fmt chunk size
        1,                           # PCM format
        _WAV_CHANNELS,
        _WAV_SAMPLE_RATE,
        _WAV_SAMPLE_RATE * _WAV_CHANNELS * _WAV_BITS_PER_SAMPLE // 8,
        _WAV_CHANNELS * _WAV_BITS_PER_SAMPLE // 8,
        _WAV_BITS_PER_SAMPLE,
        b"data",
        data_size,
    )
    return header + pcm


def create_admin_router(
    *,
    config: EngineConfig,
    tts_providers: dict,
    auth_dependency=None,
) -> APIRouter:
    """Create an APIRouter with admin endpoints.

    Args:
        config: Shared EngineConfig instance for live settings.
        tts_providers: Dict mapping engine name -> TTSProvider instance.
        auth_dependency: Optional FastAPI dependency for POST endpoints.

    Returns:
        An APIRouter to mount with ``app.include_router()``.
    """
    router = APIRouter()

    # Build dependency list for protected endpoints
    post_deps = [Depends(auth_dependency)] if auth_dependency else []

    @router.get("/admin", response_class=HTMLResponse)
    async def admin_page():
        """Serve the self-contained admin panel."""
        html_path = _STATIC_DIR / "admin.html"
        return HTMLResponse(html_path.read_text())

    @router.get("/api/config")
    async def get_config():
        """Return current engine configuration."""
        return config.snapshot()

    @router.post("/api/config", dependencies=post_deps)
    async def post_config(request: Request):
        """Apply a partial config update. Returns full config after update."""
        body = await request.json()
        try:
            return config.update(body)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/api/voices")
    async def list_voices():
        """Return available voices grouped by TTS provider."""
        result = {}
        for name, provider in tts_providers.items():
            result[name] = provider.list_voices()
        return result

    @router.post("/api/tts/preview", dependencies=post_deps)
    async def tts_preview(request: Request):
        """Synthesize text and return a WAV file for in-browser playback."""
        body = await request.json()
        text = body.get("text", "")
        engine = body.get("engine")
        voice = body.get("voice", "")

        if not engine:
            raise HTTPException(status_code=400, detail="'engine' is required")
        if engine not in tts_providers:
            raise HTTPException(status_code=404, detail=f"Unknown engine: {engine!r}")

        provider = tts_providers[engine]
        pcm = provider.synthesize(text, voice)
        wav = _build_wav(pcm)

        return Response(content=wav, media_type="audio/wav")

    return router
