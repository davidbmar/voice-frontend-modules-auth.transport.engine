"""Voice app with a custom TTS provider (swapping out the starter).

Shows how to implement the TTSProvider ABC with a production provider.
"""

from fastapi import FastAPI, WebSocket

from transport.turn import TwilioTURN
from transport.signaling import SignalingServer
from transport.session import WebRTCSession
from engine_starter.interfaces import TTSProvider
from engine_starter.stt import StarterSTT
from engine_starter.llm import StarterLLM

app = FastAPI()
stt = StarterSTT()
llm = StarterLLM()


class MyProductionTTS(TTSProvider):
    """Example: implement TTSProvider with your preferred TTS service."""

    def synthesize(self, text: str, voice: str = "") -> bytes:
        # YOUR CODE HERE: call your TTS API, return 48kHz int16 PCM bytes
        raise NotImplementedError("Replace with your TTS implementation")

    def list_voices(self) -> list[str]:
        return ["voice-1", "voice-2"]


tts = MyProductionTTS()


async def handle_call(session: WebRTCSession):
    await session.speak("Hello from custom engine!", tts)
    async for utterance in session.listen(stt):
        response = await llm.chat([{"role": "user", "content": utterance}])
        await session.speak(response, tts)


signaling = SignalingServer(turn_provider=TwilioTURN(), on_session=handle_call)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await signaling.handle(websocket)
