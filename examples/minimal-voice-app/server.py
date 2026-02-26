"""Minimal voice app — working call in ~40 lines.

Run: uvicorn server:app --port 8090
Open: http://localhost:8090
"""

import os
import time
import logging
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

app = FastAPI()

# Serve the JS client from the transport package
_repo_root = Path(__file__).resolve().parent.parent.parent
app.mount("/js", StaticFiles(directory=_repo_root / "packages" / "transport" / "js"), name="js")
stt, tts = StarterSTT(), StarterTTS()
llm = StarterLLM(model="phi3:mini")

# Use Twilio TURN if credentials are set, otherwise STUN-only (works on localhost)
if os.getenv("TWILIO_ACCOUNT_SID"):
    turn = TwilioTURN()
else:
    turn = StaticICE([{"urls": "stun:stun.l.google.com:19302"}])


async def handle_call(session: WebRTCSession):
    session.vad_silence_gap = 8  # 0.8s silence to end utterance (default 15 = 1.5s)

    t0 = time.perf_counter()
    await session.speak("Hi! How can I help?", tts)
    log.info("TIMING greeting TTS: %.2fs", time.perf_counter() - t0)

    async for utterance in session.listen(stt):
        log.info("TIMING heard utterance: %r", utterance)
        t1 = time.perf_counter()

        t_llm = time.perf_counter()
        response = await llm.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": utterance},
        ])
        log.info("TIMING LLM: %.2fs (%d chars)", time.perf_counter() - t_llm, len(response))

        t_tts = time.perf_counter()
        await session.speak(response, tts)
        log.info("TIMING TTS: %.2fs", time.perf_counter() - t_tts)

        log.info("TIMING total response: %.2fs", time.perf_counter() - t1)


signaling = SignalingServer(turn_provider=turn, on_session=handle_call)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await signaling.handle(websocket)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse("index.html")
