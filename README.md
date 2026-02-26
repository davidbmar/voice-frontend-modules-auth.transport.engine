# voice-frontend-modules — auth . transport . engine

Shared infrastructure for voice AI applications. Three independent, composable packages.

## Packages

| Package | Purpose | Status |
|---------|---------|--------|
| **transport** | TURN credentials, ICE gathering, WebRTC signaling, Cloudflare tunnel, audio utils, browser JS client | Design |
| **edge-auth** | Pluggable auth providers (Cloudflare Access, Google JWT, Bearer token) for HTTP + WebSocket | Design |
| **engine-starter** | Reference STT/TTS/LLM implementation for testing. **Not production — swap for your own providers.** | Design |

## Architecture

```
                    ┌─────────────────────────────┐
                    │        Your Voice App        │
                    │   (FSM, business logic,      │
                    │    conversation design)       │
                    └──────┬──────────┬────────────┘
                           │          │
              ┌────────────▼──┐  ┌────▼───────────┐
              │   Transport   │  │   Edge Auth     │
              │               │  │                 │
              │  TURN         │  │  CF Access      │
              │  Signaling    │  │  Google JWT     │
              │  Session      │  │  Bearer Token   │
              │  Tunnel       │  │  No Auth (dev)  │
              │  Audio utils  │  │                 │
              │  JS client    │  │                 │
              └──────┬────────┘  └─────────────────┘
                     │
              ┌──────▼────────┐
              │ Engine Starter │  ← "Swap me for production"
              │               │
              │  STT (whisper) │
              │  TTS (piper)   │
              │  LLM (ollama)  │
              │               │
              │  Defines ABCs  │
              │  that prod     │
              │  providers     │
              │  implement     │
              └───────────────┘
```

**Rule:** Transport never imports Edge Auth. Edge Auth never imports Transport. Either works alone. Engine Starter is optional — transport works with any STT/TTS/LLM that implements the ABCs.

## Quick Start

```bash
pip install voice-frontend-transport                  # just connectivity
pip install voice-frontend-edge-auth                  # just auth
pip install voice-frontend-engine-starter             # starter engine
pip install voice-frontend[all]                       # everything
```

## Minimal Voice App (~50 lines)

```python
from fastapi import FastAPI, WebSocket
from transport.turn import TwilioTURN
from transport.signaling import SignalingServer
from transport.session import WebRTCSession
from engine_starter import StarterSTT, StarterTTS, StarterLLM

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
async def ws(websocket: WebSocket):
    await signaling.handle(websocket)
```

## Consumer Projects

- [voice-calendar-scheduler-FSM](https://github.com/davidbmar/voice-calendar-scheduler-FSM) — 8-step FSM for scheduling apartment viewings via voice
- [iphone-and-companion-transcribe-mode](https://github.com/davidbmar/iphone-and-companion-transcribe-mode) — Voice transcription companion app

## Docs

- [Design Document](docs/design.md) — Full architecture, interfaces, and migration plan
