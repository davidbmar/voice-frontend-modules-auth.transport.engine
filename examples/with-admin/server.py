"""Voice app with admin panel, barge-in, and optional Kokoro TTS.

Demonstrates:
- EngineConfig for live-tunable settings
- Admin panel mounted via create_admin_router()
- speak_with_barge_in() for interruptible playback
- Piper + optional Kokoro TTS (Kokoro loaded if kokoro-onnx is installed)
- Config applied to session each turn (changes take effect immediately)

Run:  uvicorn server:app --port 8090
Open: http://localhost:8090
Admin: http://localhost:8090/admin
"""

import os
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice-app")

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from transport.turn import TwilioTURN, StaticICE
from transport.signaling import SignalingServer
from transport.session import WebRTCSession
from engine_starter.stt import StarterSTT
from engine_starter.tts import StarterTTS
from engine_starter.llm import StarterLLM
from engine_starter.config import EngineConfig
from engine_starter.admin import create_admin_router

app = FastAPI()

# ── Config ──
config = EngineConfig()

# ── TTS providers ──
tts_providers = {"piper": StarterTTS()}

try:
    import kokoro_onnx  # verify the runtime dependency is actually installed
    from engine_starter.kokoro_tts import KokoroTTS
    tts_providers["kokoro"] = KokoroTTS()
    config["tts_engine"] = "kokoro"  # default to kokoro if available
    log.info("Kokoro TTS loaded — set as default engine")
except (ImportError, FileNotFoundError) as exc:
    log.info("Kokoro TTS not available (%s) — using Piper only", exc)

# ── Admin panel ──
app.include_router(create_admin_router(config=config, tts_providers=tts_providers))

# ── Static assets ──
_repo_root = Path(__file__).resolve().parent.parent.parent
app.mount("/js", StaticFiles(directory=_repo_root / "packages" / "transport" / "js"), name="js")

stt = StarterSTT()
llm = StarterLLM(model="phi3:mini")

# ── ICE/TURN ──
if os.getenv("TWILIO_ACCOUNT_SID"):
    turn = TwilioTURN()
else:
    turn = StaticICE([{"urls": "stun:stun.l.google.com:19302"}])


def _apply_config(session: WebRTCSession):
    """Push current config values onto the session (called each turn)."""
    snap = config.snapshot()
    session.vad_energy_threshold = snap["vad_energy_threshold"]
    session.vad_speech_confirm_frames = snap["vad_speech_confirm_frames"]
    session.vad_silence_gap = snap["vad_silence_gap"]
    session.barge_in_enabled = snap["barge_in_enabled"]
    session.barge_in_energy_threshold = snap["barge_in_energy_threshold"]
    session.barge_in_confirm_frames = snap["barge_in_confirm_frames"]


def _get_tts():
    """Return the currently selected TTS provider and voice."""
    engine = config["tts_engine"]
    voice = config["tts_voice"]
    provider = tts_providers.get(engine, tts_providers.get("piper"))
    return provider, voice


async def handle_call(session: WebRTCSession):
    _apply_config(session)
    tts, voice = _get_tts()

    t0 = time.perf_counter()
    dur, interruption = await session.speak_with_barge_in(
        "Hi! How can I help?", tts, voice=voice
    )
    log.info("TIMING greeting: %.2fs (interrupted: %s)", time.perf_counter() - t0, interruption is not None)

    async for utterance in session.listen(stt):
        # Re-apply config each turn (admin panel changes take effect here)
        _apply_config(session)
        tts, voice = _get_tts()

        log.info("Heard: %r", utterance)
        t1 = time.perf_counter()

        response = await llm.chat([
            {"role": "system", "content": "You are a helpful voice assistant. Keep responses brief."},
            {"role": "user", "content": utterance},
        ])
        log.info("TIMING LLM: %.2fs", time.perf_counter() - t1)

        t_tts = time.perf_counter()
        dur, interruption = await session.speak_with_barge_in(
            response, tts, voice=voice, stt=stt
        )
        log.info("TIMING TTS: %.2fs (barge-in: %s)", time.perf_counter() - t_tts, interruption is not None)


signaling = SignalingServer(turn_provider=turn, on_session=handle_call)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await signaling.handle(websocket)


@app.get("/")
async def index():
    return FileResponse("index.html")
