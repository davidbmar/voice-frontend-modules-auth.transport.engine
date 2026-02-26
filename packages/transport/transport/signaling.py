"""WebSocket signaling server for SDP offer/answer exchange.

Handles the hello -> ICE servers -> offer/answer -> call lifecycle.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

from transport.turn import TURNProvider
from transport.session import WebRTCSession

log = logging.getLogger("transport.signaling")


class SignalingServer:
    """WebSocket signaling server for WebRTC voice calls.

    Usage with FastAPI:
        signaling = SignalingServer(turn_provider=TwilioTURN(), on_session=handle_call)

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await signaling.handle(websocket)
    """

    def __init__(
        self,
        turn_provider: TURNProvider,
        on_session: Callable[[WebRTCSession, Any], Awaitable[None] | None],
    ):
        self._turn_provider = turn_provider
        self._on_session = on_session

    async def handle(self, websocket, user=None):
        """Handle a WebSocket connection through the signaling lifecycle."""
        await websocket.accept()

        session = None
        ice_servers = []

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "hello":
                    ice_servers = await self._turn_provider.fetch_ice_servers()
                    await websocket.send_json({
                        "type": "hello_ack",
                        "ice_servers": ice_servers,
                    })

                elif msg_type == "webrtc_offer":
                    sdp = msg.get("sdp", "")
                    if not sdp:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Missing SDP in offer",
                        })
                        continue

                    try:
                        session = WebRTCSession(ice_servers=ice_servers)
                        answer_sdp = await session.handle_offer(sdp)
                        await websocket.send_json({
                            "type": "webrtc_answer",
                            "sdp": answer_sdp,
                        })
                        callback_result = self._on_session(session, websocket)
                        if asyncio.iscoroutine(callback_result):
                            asyncio.create_task(callback_result)
                    except ImportError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "WebRTC not available. Install: pip install aiortc av",
                        })
                    except Exception as e:
                        log.error("Failed to handle offer: %s", e)
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e),
                        })

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg_type == "hangup":
                    log.info("Client hung up")
                    break

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })

        except Exception:
            log.info("WebSocket disconnected")
        finally:
            if session:
                await session.close()
