"""Minimal voice app — working call in ~40 lines.

Run: uvicorn server:app --port 8090
Open: http://localhost:8090
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from transport.turn import TwilioTURN
from transport.signaling import SignalingServer
from transport.session import WebRTCSession
from engine_starter.stt import StarterSTT
from engine_starter.tts import StarterTTS
from engine_starter.llm import StarterLLM

app = FastAPI()
stt, tts, llm = StarterSTT(), StarterTTS(), StarterLLM()


async def handle_call(session: WebRTCSession):
    await session.speak("Hi! How can I help?", tts)
    async for utterance in session.listen(stt):
        response = await llm.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": utterance},
        ])
        await session.speak(response, tts)


signaling = SignalingServer(turn_provider=TwilioTURN(), on_session=handle_call)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await signaling.handle(websocket)


@app.get("/")
async def index():
    return FileResponse("index.html")
