"""Voice app with Cloudflare Access authentication.

Run: uvicorn server:app --port 8090
"""

import os

from fastapi import FastAPI, WebSocket

from transport.turn import TwilioTURN
from transport.signaling import SignalingServer
from transport.session import WebRTCSession
from edge_auth import auth_middleware, CompositeProvider
from edge_auth.providers import CloudflareAccessProvider, NoAuthProvider
from engine_starter.stt import StarterSTT
from engine_starter.tts import StarterTTS
from engine_starter.llm import StarterLLM

DEBUG = os.getenv("DEBUG", "").lower() == "true"

app = FastAPI()
stt, tts, llm = StarterSTT(), StarterTTS(), StarterLLM()

auth = CompositeProvider([
    CloudflareAccessProvider(team_domain="myteam.cloudflareaccess.com"),
    NoAuthProvider() if DEBUG else None,
])

app.add_middleware(auth_middleware(auth, exclude=["/health", "/ws"]))


async def handle_call(session: WebRTCSession):
    await session.speak("Authenticated! How can I help?", tts)
    async for utterance in session.listen(stt):
        response = await llm.chat([{"role": "user", "content": utterance}])
        await session.speak(response, tts)


signaling = SignalingServer(turn_provider=TwilioTURN(), on_session=handle_call)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    user = await auth.authenticate_ws(websocket)
    if not user.authenticated:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    await signaling.handle(websocket, user=user)


@app.get("/health")
async def health():
    return {"status": "ok"}
