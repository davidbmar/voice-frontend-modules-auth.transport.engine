# Voice Frontend Platform — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract shared WebRTC voice infrastructure from two existing projects into three reusable packages: transport, edge-auth, and engine-starter.

**Architecture:** Monorepo with three independent Python packages plus a vanilla JS client. Transport handles WebRTC connectivity (TURN, signaling, session, tunnel, audio). Edge Auth handles pluggable authentication. Engine Starter provides reference STT/TTS/LLM implementations with ABCs for production providers.

**Tech Stack:** Python 3.11+, FastAPI, aiortc, aiohttp, numpy, scipy, faster-whisper, piper-tts, pytest, pytest-asyncio

**Source repos to extract from:**
- Scheduler: `/Users/davidmar/src/voice-calendar-scheduler-FSM/`
- Companion: `/Users/davidmar/src/iphone-and-companion-transcribe-mode/`

---

## Task 1: Project Scaffolding — pyproject.toml and __init__.py files

**Files:**
- Create: `pyproject.toml` (root workspace)
- Create: `packages/transport/pyproject.toml`
- Create: `packages/transport/transport/__init__.py`
- Create: `packages/edge-auth/pyproject.toml`
- Create: `packages/edge-auth/edge_auth/__init__.py`
- Create: `packages/edge-auth/edge_auth/providers/__init__.py`
- Create: `packages/engine-starter/pyproject.toml`
- Create: `packages/engine-starter/engine_starter/__init__.py`

**Step 1: Create root pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "voice-frontend"
version = "0.1.0"
description = "Shared voice WebRTC infrastructure — auth, transport, engine"
requires-python = ">=3.11"
dependencies = [
    "voice-frontend-transport",
    "voice-frontend-edge-auth",
    "voice-frontend-engine-starter",
]

[project.optional-dependencies]
all = [
    "voice-frontend-transport",
    "voice-frontend-edge-auth",
    "voice-frontend-engine-starter",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]
```

**Step 2: Create packages/transport/pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "voice-frontend-transport"
version = "0.1.0"
description = "WebRTC transport: TURN credentials, signaling, session, tunnel, audio utilities"
requires-python = ">=3.11"
dependencies = [
    "aiohttp>=3.9,<4",
    "aiortc>=1.9,<2",
    "av>=12.0,<13",
    "numpy>=1.24,<2",
    "fastapi>=0.110,<1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

**Step 3: Create packages/edge-auth/pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "voice-frontend-edge-auth"
version = "0.1.0"
description = "Pluggable edge authentication for voice endpoints"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110,<1",
]

[project.optional-dependencies]
cloudflare = ["PyJWT>=2.8,<3", "cryptography>=42.0"]
google = ["google-auth>=2.20,<3"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

**Step 4: Create packages/engine-starter/pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "voice-frontend-engine-starter"
version = "0.1.0"
description = "Reference STT/TTS/LLM for testing — swap for production providers"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.24,<2",
    "scipy>=1.11,<2",
]

[project.optional-dependencies]
stt = ["faster-whisper>=1.0,<2"]
tts = ["piper-tts>=1.2,<2"]
llm-ollama = ["httpx>=0.27,<1"]
llm-claude = ["anthropic>=0.40,<1"]
llm-openai = ["openai>=1.30,<2"]
all = [
    "faster-whisper>=1.0,<2",
    "piper-tts>=1.2,<2",
    "httpx>=0.27,<1",
    "anthropic>=0.40,<1",
    "openai>=1.30,<2",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

**Step 5: Create __init__.py files**

`packages/transport/transport/__init__.py`:
```python
"""Voice Frontend Transport — WebRTC connectivity for voice applications."""
```

`packages/edge-auth/edge_auth/__init__.py`:
```python
"""Voice Frontend Edge Auth — pluggable authentication providers."""

from edge_auth.base import AuthProvider, AuthResult
from edge_auth.middleware import auth_middleware, auth_dependency
from edge_auth.composite import CompositeProvider

__all__ = [
    "AuthProvider",
    "AuthResult",
    "auth_middleware",
    "auth_dependency",
    "CompositeProvider",
]
```

`packages/edge-auth/edge_auth/providers/__init__.py`:
```python
"""Auth provider implementations."""

from edge_auth.providers.cloudflare import CloudflareAccessProvider
from edge_auth.providers.google import GoogleJWTProvider
from edge_auth.providers.bearer import BearerTokenProvider
from edge_auth.providers.none import NoAuthProvider

__all__ = [
    "CloudflareAccessProvider",
    "GoogleJWTProvider",
    "BearerTokenProvider",
    "NoAuthProvider",
]
```

`packages/engine-starter/engine_starter/__init__.py`:
```python
"""Voice Frontend Engine Starter — reference STT/TTS/LLM implementations.

WARNING: This is a starter engine for testing, not production.
Swap providers via the ABCs in engine_starter.interfaces.
"""

from engine_starter.interfaces import STTProvider, TTSProvider, LLMProvider, TranscriptionResult

__all__ = [
    "STTProvider",
    "TTSProvider",
    "LLMProvider",
    "TranscriptionResult",
]
```

**Step 6: Verify project structure**

Run: `find packages -name "*.toml" -o -name "__init__.py" | sort`
Expected: All files listed.

**Step 7: Commit**

```bash
git add pyproject.toml packages/*/pyproject.toml packages/*/**/__init__.py
git commit -m "feat: add pyproject.toml and package init files for monorepo"
```

---

## Task 2: transport.audio — AudioQueue and resample utilities

**Files:**
- Create: `packages/transport/transport/audio.py`
- Create: `packages/transport/tests/test_audio.py`

**Source:** Extract from `voice-calendar-scheduler-FSM/engine-repo/gateway/audio/audio_queue.py` and `iphone-and-companion-transcribe-mode/scheduling/channels/webrtc_channel.py` (resample logic).

**Step 1: Write failing tests**

```python
# packages/transport/tests/test_audio.py
"""Tests for transport.audio — AudioQueue and resample utilities."""

import numpy as np
import pytest

from transport.audio import AudioQueue, resample


class TestAudioQueue:
    def test_read_empty_returns_silence(self):
        q = AudioQueue()
        data = q.read(100)
        assert len(data) == 100
        assert data == b"\x00" * 100

    def test_enqueue_and_read_exact(self):
        q = AudioQueue()
        q.enqueue(b"\x01\x02\x03\x04")
        assert q.read(4) == b"\x01\x02\x03\x04"

    def test_read_pads_silence_when_not_enough_data(self):
        q = AudioQueue()
        q.enqueue(b"\xff\xff")
        data = q.read(4)
        assert data == b"\xff\xff\x00\x00"

    def test_read_spans_multiple_chunks(self):
        q = AudioQueue()
        q.enqueue(b"\x01\x02")
        q.enqueue(b"\x03\x04")
        assert q.read(4) == b"\x01\x02\x03\x04"

    def test_clear_discards_all(self):
        q = AudioQueue()
        q.enqueue(b"\x01\x02\x03\x04")
        q.clear()
        assert q.available == 0
        assert q.read(4) == b"\x00\x00\x00\x00"

    def test_available_property(self):
        q = AudioQueue()
        assert q.available == 0
        q.enqueue(b"\x01\x02")
        assert q.available == 2
        q.read(1)
        assert q.available == 1

    def test_enqueue_empty_is_noop(self):
        q = AudioQueue()
        q.enqueue(b"")
        assert q.available == 0


class TestResample:
    def test_downsample_48k_to_16k(self):
        # 960 samples at 48kHz = 20ms → should become 320 samples at 16kHz
        samples_48k = np.zeros(960, dtype=np.int16)
        samples_48k[0] = 1000
        result = resample(samples_48k.tobytes(), from_rate=48000, to_rate=16000)
        result_arr = np.frombuffer(result, dtype=np.int16)
        assert len(result_arr) == 320

    def test_upsample_16k_to_48k(self):
        samples_16k = np.zeros(320, dtype=np.int16)
        result = resample(samples_16k.tobytes(), from_rate=16000, to_rate=48000)
        result_arr = np.frombuffer(result, dtype=np.int16)
        assert len(result_arr) == 960

    def test_same_rate_is_passthrough(self):
        pcm = np.array([100, 200, 300], dtype=np.int16).tobytes()
        result = resample(pcm, from_rate=48000, to_rate=48000)
        assert result == pcm
```

**Step 2: Run tests to verify they fail**

Run: `cd packages/transport && python -m pytest tests/test_audio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transport.audio'`

**Step 3: Implement transport/audio.py**

```python
# packages/transport/transport/audio.py
"""Audio utilities for WebRTC voice transport.

AudioQueue: Thread-safe FIFO for TTS output (never drops audio).
resample: Convert PCM between sample rates (e.g., 48kHz WebRTC ↔ 16kHz STT).
"""

import threading
from collections import deque

import numpy as np


class AudioQueue:
    """Unbounded FIFO of PCM byte blobs, read out in fixed-size chunks.

    Producers call enqueue() with variable-length PCM blobs (one per TTS sentence).
    The consumer calls read(n) every 20ms to get exactly n bytes, zero-padded
    if not enough data is available yet.
    """

    def __init__(self):
        self._chunks: deque[bytes] = deque()
        self._current = b""
        self._offset = 0
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        """Total bytes available for reading."""
        with self._lock:
            total = len(self._current) - self._offset
            for chunk in self._chunks:
                total += len(chunk)
            return total

    def enqueue(self, data: bytes):
        """Append a PCM blob to the queue (thread-safe)."""
        if not data:
            return
        with self._lock:
            self._chunks.append(data)

    def read(self, n: int) -> bytes:
        """Read exactly n bytes. Zero-pads if not enough data available."""
        with self._lock:
            result = bytearray(n)
            written = 0

            while written < n:
                if self._offset >= len(self._current):
                    if not self._chunks:
                        break
                    self._current = self._chunks.popleft()
                    self._offset = 0

                remaining_in_chunk = len(self._current) - self._offset
                to_copy = min(remaining_in_chunk, n - written)
                result[written:written + to_copy] = self._current[self._offset:self._offset + to_copy]
                self._offset += to_copy
                written += to_copy

            return bytes(result)

    def clear(self):
        """Discard all queued audio."""
        with self._lock:
            self._chunks.clear()
            self._current = b""
            self._offset = 0


def resample(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample int16 PCM between sample rates.

    Args:
        pcm_bytes: Raw PCM int16 mono audio bytes.
        from_rate: Source sample rate (e.g., 48000).
        to_rate: Target sample rate (e.g., 16000).

    Returns:
        Resampled PCM int16 bytes.
    """
    if from_rate == to_rate:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
    num_output = int(len(samples) * to_rate / from_rate)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, num_output),
        np.arange(len(samples)),
        samples,
    )
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()
```

**Step 4: Run tests to verify they pass**

Run: `cd packages/transport && python -m pytest tests/test_audio.py -v`
Expected: All 10 tests PASS.

**Step 5: Commit**

```bash
git add packages/transport/transport/audio.py packages/transport/tests/test_audio.py
git commit -m "feat(transport): add AudioQueue and resample utilities"
```

---

## Task 3: transport.turn — TURN credential provider with ABC

**Files:**
- Create: `packages/transport/transport/turn.py`
- Create: `packages/transport/tests/test_turn.py`

**Source:** Extract from `voice-calendar-scheduler-FSM/gateway/turn.py`.

**Step 1: Write failing tests**

```python
# packages/transport/tests/test_turn.py
"""Tests for transport.turn — TURN credential providers."""

import pytest

from transport.turn import TURNProvider, TwilioTURN, StaticICE


class TestStaticICE:
    @pytest.mark.asyncio
    async def test_returns_configured_servers(self):
        servers = [{"urls": "stun:stun.example.com:3478"}]
        provider = StaticICE(servers=servers)
        result = await provider.fetch_ice_servers()
        assert result == servers

    @pytest.mark.asyncio
    async def test_empty_list(self):
        provider = StaticICE(servers=[])
        result = await provider.fetch_ice_servers()
        assert result == []


class TestTwilioTURN:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        provider = TwilioTURN()
        result = await provider.fetch_ice_servers()
        assert result == []

    def test_is_turn_provider(self):
        assert issubclass(TwilioTURN, TURNProvider)
        assert issubclass(StaticICE, TURNProvider)
```

**Step 2: Run tests to verify they fail**

Run: `cd packages/transport && python -m pytest tests/test_turn.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement transport/turn.py**

```python
# packages/transport/transport/turn.py
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
        """Return ICE server dicts in WebRTC format.

        Each dict has: {"urls": "turn:...", "username": "...", "credential": "..."}
        """
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
```

**Step 4: Run tests**

Run: `cd packages/transport && python -m pytest tests/test_turn.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add packages/transport/transport/turn.py packages/transport/tests/test_turn.py
git commit -m "feat(transport): add TURN provider ABC with Twilio and StaticICE implementations"
```

---

## Task 4: transport.audio_source — WebRTC audio track for aiortc

**Files:**
- Create: `packages/transport/transport/audio_source.py`
- Create: `packages/transport/tests/test_audio_source.py`

**Source:** Extract from `voice-calendar-scheduler-FSM/engine-repo/gateway/audio/webrtc_audio_source.py`.

**Step 1: Write failing tests**

```python
# packages/transport/tests/test_audio_source.py
"""Tests for transport.audio_source — WebRTC audio track."""

import numpy as np
import pytest

from transport.audio_source import WebRTCAudioSource, FRAME_SAMPLES, SAMPLE_RATE


class TestWebRTCAudioSource:
    @pytest.mark.asyncio
    async def test_recv_returns_silence_without_generator(self):
        source = WebRTCAudioSource()
        frame = await source.recv()
        assert frame.sample_rate == SAMPLE_RATE
        arr = frame.to_ndarray().flatten()
        assert len(arr) == FRAME_SAMPLES
        assert np.all(arr == 0)

    @pytest.mark.asyncio
    async def test_recv_returns_generator_audio(self):
        source = WebRTCAudioSource()

        class FakeGen:
            def next_chunk(self):
                from engine_starter.interfaces import AudioChunk
                pcm = np.ones(FRAME_SAMPLES, dtype=np.int16).tobytes()
                return AudioChunk(samples=pcm, sample_rate=SAMPLE_RATE, channels=1)

        source.set_generator(FakeGen())
        frame = await source.recv()
        arr = frame.to_ndarray().flatten()
        assert np.all(arr == 1)

    @pytest.mark.asyncio
    async def test_clear_generator_reverts_to_silence(self):
        source = WebRTCAudioSource()

        class FakeGen:
            def next_chunk(self):
                from engine_starter.interfaces import AudioChunk
                pcm = np.ones(FRAME_SAMPLES, dtype=np.int16).tobytes()
                return AudioChunk(samples=pcm, sample_rate=SAMPLE_RATE, channels=1)

        source.set_generator(FakeGen())
        await source.recv()  # generator audio
        source.clear_generator()
        frame = await source.recv()
        arr = frame.to_ndarray().flatten()
        assert np.all(arr == 0)
```

**Step 2: Run tests to verify they fail**

Run: `cd packages/transport && python -m pytest tests/test_audio_source.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement transport/audio_source.py**

```python
# packages/transport/transport/audio_source.py
"""Custom audio track for aiortc that serves PCM frames over WebRTC.

aiortc calls recv() roughly every 20ms. We return silence or pull
a chunk from the attached audio generator (TTS output).
"""

import asyncio
import time
from fractions import Fraction

import numpy as np
from av import AudioFrame
from aiortc import MediaStreamTrack

SAMPLE_RATE = 48000
FRAME_SAMPLES = 960  # 20ms at 48kHz
PTIME = FRAME_SAMPLES / SAMPLE_RATE  # 0.02 seconds


class WebRTCAudioSource(MediaStreamTrack):
    """Server-side audio track — streams silence or generator output."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._generator = None
        self._start_time = None
        self._frame_count = 0

    def set_generator(self, generator):
        """Attach an audio generator (must have a next_chunk() method)."""
        self._generator = generator

    def clear_generator(self):
        """Detach the generator — track reverts to silence."""
        self._generator = None

    async def recv(self) -> AudioFrame:
        """Called by aiortc to get the next audio frame."""
        if self._start_time is None:
            self._start_time = time.monotonic()

        target_time = self._start_time + self._frame_count * PTIME
        now = time.monotonic()
        if target_time > now:
            await asyncio.sleep(target_time - now)

        self._frame_count += 1

        if self._generator:
            chunk = self._generator.next_chunk()
            samples = np.frombuffer(chunk.samples, dtype=np.int16)
        else:
            samples = np.zeros(FRAME_SAMPLES, dtype=np.int16)

        frame = AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = SAMPLE_RATE
        frame.pts = (self._frame_count - 1) * FRAME_SAMPLES
        frame.time_base = Fraction(1, SAMPLE_RATE)

        return frame
```

**Step 4: Run tests**

Run: `cd packages/transport && python -m pytest tests/test_audio_source.py -v`
Expected: PASS (may need engine_starter.interfaces first — see Task 7 for AudioChunk; create a minimal stub if needed).

**Step 5: Commit**

```bash
git add packages/transport/transport/audio_source.py packages/transport/tests/test_audio_source.py
git commit -m "feat(transport): add WebRTCAudioSource track for aiortc"
```

---

## Task 5: transport.session — WebRTC peer connection lifecycle

**Files:**
- Create: `packages/transport/transport/session.py`
- Create: `packages/transport/tests/test_session.py`

**Source:** Extract from `voice-calendar-scheduler-FSM/engine-repo/gateway/webrtc.py` (Session class).

**Step 1: Write failing tests**

```python
# packages/transport/tests/test_session.py
"""Tests for transport.session — WebRTC session lifecycle."""

import re
import pytest

from transport.session import WebRTCSession


class TestCleanForSpeech:
    def test_strips_markdown_headers(self):
        assert WebRTCSession._clean_for_speech("## Hello World") == "Hello World"

    def test_strips_bold_and_italic(self):
        assert "asterisk" not in WebRTCSession._clean_for_speech("**bold** and *italic*").lower()
        assert "bold" in WebRTCSession._clean_for_speech("**bold** and *italic*")

    def test_strips_bullet_points(self):
        text = "- Item one\n- Item two"
        result = WebRTCSession._clean_for_speech(text)
        assert result.startswith("Item one")
        assert "-" not in result

    def test_strips_urls(self):
        text = "Visit https://example.com for more"
        result = WebRTCSession._clean_for_speech(text)
        assert "https://" not in result

    def test_strips_inline_code(self):
        result = WebRTCSession._clean_for_speech("Run `npm install` to start")
        assert "`" not in result
        assert "npm install" in result


class TestSplitSentences:
    def test_splits_on_period(self):
        result = WebRTCSession._split_sentences("Hello. World.")
        assert result == ["Hello.", "World."]

    def test_single_sentence(self):
        result = WebRTCSession._split_sentences("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_empty_string(self):
        result = WebRTCSession._split_sentences("")
        assert result == []


class TestIceServersToRtc:
    def test_converts_urls_string(self):
        from transport.session import ice_servers_to_rtc
        servers = [{"urls": "stun:stun.example.com:3478"}]
        result = ice_servers_to_rtc(servers)
        assert len(result) == 1
        assert "stun.example.com" in str(result[0].urls)

    def test_handles_url_key(self):
        from transport.session import ice_servers_to_rtc
        servers = [{"url": "turn:turn.example.com:3478", "username": "u", "credential": "c"}]
        result = ice_servers_to_rtc(servers)
        assert len(result) == 1
```

**Step 2: Run tests to verify they fail**

Run: `cd packages/transport && python -m pytest tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement transport/session.py**

```python
# packages/transport/transport/session.py
"""WebRTC session — one active voice call.

Manages the aiortc RTCPeerConnection, mic audio capture, TTS playback,
and provides high-level speak()/listen() APIs.
"""

import asyncio
import logging
import re
from typing import AsyncIterator, Callable, Awaitable

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

from transport.audio import AudioQueue
from transport.audio_source import WebRTCAudioSource, FRAME_SAMPLES, SAMPLE_RATE

log = logging.getLogger("transport.session")


def ice_servers_to_rtc(servers: list) -> list:
    """Convert ICE server dicts to RTCIceServer objects."""
    result = []
    for s in servers:
        urls = s.get("urls", s.get("url", ""))
        if isinstance(urls, str):
            urls = [urls]
        result.append(RTCIceServer(
            urls=urls,
            username=s.get("username", ""),
            credential=s.get("credential", ""),
        ))
    return result


class QueuedGenerator:
    """Reads PCM from an AudioQueue in 20ms chunks for WebRTCAudioSource."""

    def __init__(self, queue: AudioQueue):
        self.queue = queue

    def next_chunk(self):
        from engine_starter.interfaces import AudioChunk
        pcm = self.queue.read(FRAME_SAMPLES * 2)
        return AudioChunk(samples=pcm, sample_rate=SAMPLE_RATE, channels=1)


class WebRTCSession:
    """One active WebRTC call. Provides mic audio input and TTS output."""

    def __init__(self, ice_servers: list | None = None):
        rtc_servers = ice_servers_to_rtc(ice_servers or [])
        config = RTCConfiguration(iceServers=rtc_servers) if rtc_servers else RTCConfiguration()
        self._pc = RTCPeerConnection(configuration=config)
        self._audio_source = WebRTCAudioSource()

        self._audio_queue = AudioQueue()
        self._tts_generator = QueuedGenerator(self._audio_queue)

        self._recording = False
        self._mic_frames: list[bytes] = []
        self._mic_track = None
        self._mic_recv_task: asyncio.Task | None = None

        # VAD settings (live-tunable)
        self.vad_energy_threshold: int = 500
        self.vad_silence_gap: int = 15  # polls (~1.5s at 100ms poll)
        self.vad_speech_confirm_frames: int = 2

        @self._pc.on("connectionstatechange")
        async def on_conn_state():
            log.info("Connection state: %s", self._pc.connectionState)

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_state():
            log.info("ICE connection state: %s", self._pc.iceConnectionState)

        @self._pc.on("track")
        async def on_track(track):
            if track.kind != "audio":
                return
            log.info("Received remote audio track from browser mic")
            self._mic_track = track
            self._mic_recv_task = asyncio.ensure_future(self._recv_mic_audio(track))

    async def handle_offer(self, sdp: str) -> str:
        """Process browser SDP offer, return SDP answer."""
        self._pc.addTrack(self._audio_source)
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self._pc.setRemoteDescription(offer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        log.info("SDP answer created")
        return self._pc.localDescription.sdp

    async def speak(self, text: str, tts) -> float:
        """Synthesize and play audio. Returns duration in seconds.

        Args:
            text: Text to speak.
            tts: Any object with an async synthesize(text, voice) -> bytes method,
                 or a TTSProvider instance.
        """
        self._audio_source.set_generator(self._tts_generator)
        text = self._clean_for_speech(text)
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0

        log.info("TTS: %d sentences to synthesize", len(sentences))
        loop = asyncio.get_event_loop()
        total_bytes = 0

        for i, sentence in enumerate(sentences):
            if asyncio.iscoroutinefunction(getattr(tts, 'synthesize', None)):
                pcm = await tts.synthesize(sentence)
            else:
                pcm = await loop.run_in_executor(None, tts.synthesize, sentence)
            if pcm:
                self._audio_queue.enqueue(pcm)
                total_bytes += len(pcm)

        duration = total_bytes / (SAMPLE_RATE * 2)
        log.info("TTS total: %d bytes, %.1fs playback", total_bytes, duration)
        return duration

    async def listen(self, stt) -> AsyncIterator[str]:
        """Yield transcribed utterances using energy-based VAD + STT.

        Uses configurable VAD settings: vad_energy_threshold, vad_silence_gap,
        vad_speech_confirm_frames.

        Args:
            stt: Any object with a transcribe(audio_bytes, sample_rate) method.
        """
        self._mic_frames.clear()
        self._recording = True
        log.info("Listening started")

        # Wait for mic track
        for _ in range(50):  # 5 seconds
            if self._mic_track is not None:
                break
            await asyncio.sleep(0.1)

        if self._mic_track is None:
            log.warning("No mic track received after 5s")
            self._recording = False
            return

        speech_frames = 0
        silence_frames = 0
        in_speech = False
        loop = asyncio.get_event_loop()

        try:
            while self._recording:
                await asyncio.sleep(0.1)

                if not self._mic_frames:
                    continue

                # Compute RMS of recent frames
                recent = self._mic_frames[-5:] if len(self._mic_frames) >= 5 else self._mic_frames
                pcm = b"".join(recent)
                if pcm:
                    samples = np.frombuffer(pcm, dtype=np.int16)
                    rms = int(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                else:
                    rms = 0

                if rms >= self.vad_energy_threshold:
                    speech_frames += 1
                    silence_frames = 0
                    if speech_frames >= self.vad_speech_confirm_frames:
                        in_speech = True
                else:
                    silence_frames += 1
                    speech_frames = 0

                    if in_speech and silence_frames >= self.vad_silence_gap:
                        # End of utterance — transcribe
                        in_speech = False
                        pcm_data = b"".join(self._mic_frames)
                        self._mic_frames.clear()

                        if pcm_data:
                            result = await loop.run_in_executor(
                                None, stt.transcribe, pcm_data, SAMPLE_RATE
                            )
                            text = result.text if hasattr(result, 'text') else result[0]
                            if text and text.strip():
                                yield text.strip()
        finally:
            self._recording = False
            log.info("Listening stopped")

    def stop_speaking(self):
        """Stop TTS playback — clear the audio queue."""
        self._audio_queue.clear()
        self._audio_source.clear_generator()
        log.info("TTS playback stopped")

    async def _recv_mic_audio(self, track):
        """Background: receive audio frames from browser mic track."""
        logged_format = False
        while True:
            try:
                frame = await track.recv()
            except Exception:
                log.info("Mic track ended")
                break

            if not logged_format:
                arr = frame.to_ndarray()
                log.info("Mic frame format=%s rate=%d shape=%s dtype=%s",
                         frame.format.name, frame.sample_rate, arr.shape, arr.dtype)
                logged_format = True

            if self._recording:
                arr = frame.to_ndarray()
                if arr.dtype in (np.float32, np.float64):
                    arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
                flat = arr.flatten()
                channels = flat.shape[0] // frame.samples
                if channels > 1:
                    flat = flat[::channels]
                self._mic_frames.append(flat.astype(np.int16).tobytes())

    async def close(self):
        """Tear down the peer connection."""
        self._audio_source.clear_generator()
        self._recording = False
        if self._mic_recv_task:
            self._mic_recv_task.cancel()
        await self._pc.close()
        log.info("Session closed")

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Strip markdown formatting for clean TTS output."""
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*\u2022]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
        text = re.sub(r'\*{1,3}', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        has_urls = bool(re.search(r'https?://\S+', text))
        text = re.sub(r'(?:visit|check|see|at|on|from)\s+https?://\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = text.strip()
        if has_urls and text:
            text += " See the links on screen for more details."
        return text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for incremental TTS."""
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p for p in parts if p.strip()]
```

**Step 4: Run tests**

Run: `cd packages/transport && python -m pytest tests/test_session.py -v`
Expected: Static method tests PASS. (Integration tests need a real aiortc setup.)

**Step 5: Commit**

```bash
git add packages/transport/transport/session.py packages/transport/tests/test_session.py
git commit -m "feat(transport): add WebRTCSession with speak/listen/VAD"
```

---

## Task 6: transport.signaling — WebSocket signaling server

**Files:**
- Create: `packages/transport/transport/signaling.py`
- Create: `packages/transport/tests/test_signaling.py`

**Source:** Extract from `voice-calendar-scheduler-FSM/gateway/server.py:handle_signaling_ws()`.

**Step 1: Write failing tests**

```python
# packages/transport/tests/test_signaling.py
"""Tests for transport.signaling — WebSocket signaling server."""

import json
import pytest

from transport.signaling import SignalingServer
from transport.turn import StaticICE


class FakeWebSocket:
    """Minimal mock for FastAPI WebSocket."""

    def __init__(self, messages: list[str]):
        self._messages = list(messages)
        self._sent: list[dict] = []
        self._accepted = False
        self._closed = False

    async def accept(self):
        self._accepted = True

    async def receive_text(self) -> str:
        if not self._messages:
            from starlette.websockets import WebSocketDisconnect
            raise WebSocketDisconnect(code=1000)
        return self._messages.pop(0)

    async def send_json(self, data: dict):
        self._sent.append(data)

    async def close(self, code: int = 1000, reason: str = ""):
        self._closed = True


class TestSignalingServer:
    @pytest.mark.asyncio
    async def test_hello_returns_ice_servers(self):
        ice = [{"urls": "stun:stun.example.com:3478"}]
        server = SignalingServer(
            turn_provider=StaticICE(servers=ice),
            on_session=lambda session: None,
        )
        ws = FakeWebSocket([json.dumps({"type": "hello"})])
        await server.handle(ws)

        assert ws._accepted
        assert len(ws._sent) >= 1
        assert ws._sent[0]["type"] == "hello_ack"
        assert ws._sent[0]["ice_servers"] == ice

    @pytest.mark.asyncio
    async def test_ping_returns_pong(self):
        server = SignalingServer(
            turn_provider=StaticICE(),
            on_session=lambda session: None,
        )
        ws = FakeWebSocket([
            json.dumps({"type": "hello"}),
            json.dumps({"type": "ping"}),
        ])
        await server.handle(ws)

        types = [m["type"] for m in ws._sent]
        assert "pong" in types

    @pytest.mark.asyncio
    async def test_unknown_type_returns_error(self):
        server = SignalingServer(
            turn_provider=StaticICE(),
            on_session=lambda session: None,
        )
        ws = FakeWebSocket([
            json.dumps({"type": "hello"}),
            json.dumps({"type": "unknown_thing"}),
        ])
        await server.handle(ws)

        error_msgs = [m for m in ws._sent if m["type"] == "error"]
        assert len(error_msgs) == 1
```

**Step 2: Run tests**

Run: `cd packages/transport && python -m pytest tests/test_signaling.py -v`
Expected: FAIL

**Step 3: Implement transport/signaling.py**

```python
# packages/transport/transport/signaling.py
"""WebSocket signaling server for SDP offer/answer exchange.

Handles the hello → ICE servers → offer/answer → call lifecycle.
"""

import asyncio
import json
import logging
from typing import Callable, Awaitable

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
        on_session: Callable[[WebRTCSession], Awaitable[None] | None],
    ):
        self._turn_provider = turn_provider
        self._on_session = on_session

    async def handle(self, websocket, user=None):
        """Handle a WebSocket connection through the signaling lifecycle.

        Args:
            websocket: A FastAPI WebSocket (or any object with accept/receive_text/send_json).
            user: Optional authenticated user info (from edge-auth).
        """
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
                        # Fire the session callback
                        callback_result = self._on_session(session)
                        if asyncio.iscoroutine(callback_result):
                            asyncio.ensure_future(callback_result)
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
            # WebSocketDisconnect or other connection errors
            log.info("WebSocket disconnected")
        finally:
            if session:
                await session.close()
```

**Step 4: Run tests**

Run: `cd packages/transport && python -m pytest tests/test_signaling.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add packages/transport/transport/signaling.py packages/transport/tests/test_signaling.py
git commit -m "feat(transport): add SignalingServer for WebSocket SDP exchange"
```

---

## Task 7: engine_starter.interfaces — ABCs and data classes

**Files:**
- Create: `packages/engine-starter/engine_starter/interfaces.py`
- Create: `packages/engine-starter/tests/test_interfaces.py`

**Step 1: Write failing tests**

```python
# packages/engine-starter/tests/test_interfaces.py
"""Tests for engine_starter.interfaces — provider ABCs."""

import pytest

from engine_starter.interfaces import (
    STTProvider,
    TTSProvider,
    LLMProvider,
    TranscriptionResult,
    AudioChunk,
)


class TestTranscriptionResult:
    def test_defaults(self):
        r = TranscriptionResult(text="hello")
        assert r.text == "hello"
        assert r.no_speech_probability == 0.0
        assert r.language == ""

    def test_with_all_fields(self):
        r = TranscriptionResult(text="hello", no_speech_probability=0.1, language="en")
        assert r.no_speech_probability == 0.1


class TestAudioChunk:
    def test_fields(self):
        chunk = AudioChunk(samples=b"\x00\x00", sample_rate=48000, channels=1)
        assert chunk.sample_rate == 48000
        assert chunk.channels == 1


class TestABCsCannotBeInstantiated:
    def test_stt_is_abstract(self):
        with pytest.raises(TypeError):
            STTProvider()

    def test_tts_is_abstract(self):
        with pytest.raises(TypeError):
            TTSProvider()

    def test_llm_is_abstract(self):
        with pytest.raises(TypeError):
            LLMProvider()
```

**Step 2: Implement**

```python
# packages/engine-starter/engine_starter/interfaces.py
"""Provider interfaces — ABCs for STT, TTS, and LLM.

These are the contracts that production providers must implement.
The starter implementations in this package satisfy them for testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    text: str
    no_speech_probability: float = 0.0
    language: str = ""


@dataclass
class AudioChunk:
    """A chunk of PCM audio data."""
    samples: bytes
    sample_rate: int
    channels: int


class STTProvider(ABC):
    """Speech-to-text: audio bytes -> text."""

    @abstractmethod
    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe PCM int16 audio to text."""
        ...


class TTSProvider(ABC):
    """Text-to-speech: text -> audio bytes."""

    @abstractmethod
    def synthesize(self, text: str, voice: str = "") -> bytes:
        """Convert text to PCM int16 audio bytes at 48kHz."""
        ...

    def list_voices(self) -> list[str]:
        """Return available voice IDs. Override to provide voice selection."""
        return []


class LLMProvider(ABC):
    """Language model: messages -> response text."""

    @abstractmethod
    async def chat(self, messages: list[dict]) -> str:
        """Generate a response from conversation messages."""
        ...
```

**Step 3: Run tests**

Run: `cd packages/engine-starter && python -m pytest tests/test_interfaces.py -v`
Expected: All 5 tests PASS.

**Step 4: Commit**

```bash
git add packages/engine-starter/engine_starter/interfaces.py packages/engine-starter/tests/test_interfaces.py
git commit -m "feat(engine-starter): add STT/TTS/LLM provider ABCs and data classes"
```

---

## Task 8: engine_starter.stt — StarterSTT (faster-whisper)

**Files:**
- Create: `packages/engine-starter/engine_starter/stt.py`
- Create: `packages/engine-starter/tests/test_stt.py`

**Source:** Extract from `iphone-and-companion-transcribe-mode/engine/stt.py`.

**Step 1: Write failing tests**

```python
# packages/engine-starter/tests/test_stt.py
"""Tests for engine_starter.stt — StarterSTT."""

import numpy as np
import pytest

from engine_starter.interfaces import STTProvider, TranscriptionResult
from engine_starter.stt import StarterSTT


class TestStarterSTT:
    def test_is_stt_provider(self):
        assert issubclass(StarterSTT, STTProvider)

    def test_transcribe_empty_returns_empty(self):
        stt = StarterSTT()
        result = stt.transcribe(b"", sample_rate=16000)
        assert result.text == ""

    def test_transcribe_silence_returns_empty_or_low_confidence(self):
        """1 second of silence at 16kHz."""
        silence = np.zeros(16000, dtype=np.int16).tobytes()
        stt = StarterSTT()
        result = stt.transcribe(silence, sample_rate=16000)
        # Silence should yield empty text or high no_speech_probability
        assert result.text == "" or result.no_speech_probability > 0.5
```

**Step 2: Implement**

```python
# packages/engine-starter/engine_starter/stt.py
"""StarterSTT — faster-whisper speech-to-text.

Downloads the 'base' model (~75MB) on first use. CPU-only, int8 quantized.
For production, implement STTProvider with Deepgram, AssemblyAI, etc.
"""

import logging

import numpy as np

from engine_starter.interfaces import STTProvider, TranscriptionResult

log = logging.getLogger("engine_starter.stt")

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    from faster_whisper import WhisperModel

    log.info("Loading faster-whisper model: base (first run downloads ~75MB)...")
    _model = WhisperModel("base", device="cpu", compute_type="int8")
    log.info("Whisper model loaded")
    return _model


class StarterSTT(STTProvider):
    """Reference STT using faster-whisper (base model, CPU, int8)."""

    def transcribe(self, audio: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        if not audio:
            return TranscriptionResult(text="")

        model = _get_model()

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy.signal import resample as scipy_resample
            num_output = int(len(samples) * 16000 / sample_rate)
            samples = scipy_resample(samples, num_output).astype(np.float32)

        segments, info = model.transcribe(samples, beam_size=5, language="en")

        text_parts = []
        worst_no_speech = 0.0
        for segment in segments:
            text_parts.append(segment.text.strip())
            worst_no_speech = max(worst_no_speech, segment.no_speech_prob)

        text = " ".join(text_parts).strip()
        log.info("Transcription: %r (no_speech=%.2f)", text[:100], worst_no_speech)
        return TranscriptionResult(text=text, no_speech_probability=worst_no_speech)
```

**Step 3: Run tests**

Run: `cd packages/engine-starter && python -m pytest tests/test_stt.py -v`
Expected: PASS (note: first run downloads model).

**Step 4: Commit**

```bash
git add packages/engine-starter/engine_starter/stt.py packages/engine-starter/tests/test_stt.py
git commit -m "feat(engine-starter): add StarterSTT with faster-whisper"
```

---

## Task 9: engine_starter.tts — StarterTTS (Piper)

**Files:**
- Create: `packages/engine-starter/engine_starter/tts.py`
- Create: `packages/engine-starter/tests/test_tts.py`

**Source:** Extract from `iphone-and-companion-transcribe-mode/engine/tts.py`.

**Step 1: Write failing tests**

```python
# packages/engine-starter/tests/test_tts.py
"""Tests for engine_starter.tts — StarterTTS."""

import pytest

from engine_starter.interfaces import TTSProvider
from engine_starter.tts import StarterTTS, VOICE_CATALOG


class TestStarterTTS:
    def test_is_tts_provider(self):
        assert issubclass(StarterTTS, TTSProvider)

    def test_list_voices_returns_catalog(self):
        tts = StarterTTS()
        voices = tts.list_voices()
        assert len(voices) > 0
        assert all(isinstance(v, str) for v in voices)

    def test_synthesize_returns_bytes(self):
        tts = StarterTTS()
        result = tts.synthesize("Hello world")
        assert isinstance(result, bytes)
        assert len(result) > 0  # Should produce audio

    def test_synthesize_empty_returns_empty(self):
        tts = StarterTTS()
        result = tts.synthesize("")
        assert result == b""


class TestVoiceCatalog:
    def test_has_default_voice(self):
        ids = [v["id"] for v in VOICE_CATALOG]
        assert "en_US-lessac-medium" in ids

    def test_all_entries_have_required_fields(self):
        for v in VOICE_CATALOG:
            assert "id" in v
            assert "name" in v
            assert "lang" in v
```

**Step 2: Implement**

```python
# packages/engine-starter/engine_starter/tts.py
"""StarterTTS — Piper ONNX text-to-speech.

Downloads voice models from HuggingFace on first use.
Outputs 48kHz mono int16 PCM (resampled from Piper's native 22050Hz).
For production, implement TTSProvider with ElevenLabs, Azure, etc.
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import resample as scipy_resample

from engine_starter.interfaces import TTSProvider

log = logging.getLogger("engine_starter.tts")

TARGET_RATE = 48000
MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"

VOICE_CATALOG = [
    {"id": "en_US-lessac-medium", "name": "Lessac (US)", "lang": "en", "locale": "en_US", "voice_name": "lessac", "quality": "medium"},
    {"id": "en_US-hfc_female-medium", "name": "HFC Female (US)", "lang": "en", "locale": "en_US", "voice_name": "hfc_female", "quality": "medium"},
    {"id": "en_US-hfc_male-medium", "name": "HFC Male (US)", "lang": "en", "locale": "en_US", "voice_name": "hfc_male", "quality": "medium"},
    {"id": "en_US-libritts_r-medium", "name": "LibriTTS (US)", "lang": "en", "locale": "en_US", "voice_name": "libritts_r", "quality": "medium"},
    {"id": "en_GB-alba-medium", "name": "Alba (UK)", "lang": "en", "locale": "en_GB", "voice_name": "alba", "quality": "medium"},
    {"id": "en_GB-aru-medium", "name": "Aru (UK)", "lang": "en", "locale": "en_GB", "voice_name": "aru", "quality": "medium"},
    {"id": "de_DE-thorsten-medium", "name": "Thorsten (German)", "lang": "de", "locale": "de_DE", "voice_name": "thorsten", "quality": "medium"},
    {"id": "fr_FR-siwis-medium", "name": "Siwis (French)", "lang": "fr", "locale": "fr_FR", "voice_name": "siwis", "quality": "medium"},
    {"id": "es_ES-davefx-medium", "name": "DaveFX (Spanish)", "lang": "es", "locale": "es_ES", "voice_name": "davefx", "quality": "medium"},
]

DEFAULT_VOICE = "en_US-lessac-medium"
_CATALOG_BY_ID = {v["id"]: v for v in VOICE_CATALOG}
_voice_cache: dict = {}


def _download_model(voice_id: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODEL_DIR / f"{voice_id}.onnx"
    config_path = MODEL_DIR / f"{voice_id}.onnx.json"
    entry = _CATALOG_BY_ID[voice_id]
    base = (
        f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{entry['lang']}/{entry['locale']}/{entry['voice_name']}/{entry['quality']}/{voice_id}"
    )
    if not onnx_path.exists():
        log.info("Downloading voice model: %s ...", voice_id)
        urllib.request.urlretrieve(f"{base}.onnx", onnx_path)
    if not config_path.exists():
        log.info("Downloading voice config: %s ...", voice_id)
        urllib.request.urlretrieve(f"{base}.onnx.json", config_path)
    return onnx_path


def _get_voice(voice_id: str = ""):
    voice_id = voice_id or DEFAULT_VOICE
    if voice_id in _voice_cache:
        return _voice_cache[voice_id]
    if voice_id not in _CATALOG_BY_ID:
        log.warning("Unknown voice %r, using default", voice_id)
        voice_id = DEFAULT_VOICE
    from piper import PiperVoice
    model_path = _download_model(voice_id)
    voice = PiperVoice.load(str(model_path))
    log.info("Piper voice loaded: %s (native rate: %dHz)", voice_id, voice.config.sample_rate)
    _voice_cache[voice_id] = voice
    return voice


class StarterTTS(TTSProvider):
    """Reference TTS using Piper ONNX. Downloads models on first use."""

    def synthesize(self, text: str, voice: str = "") -> bytes:
        if not text or not text.strip():
            return b""

        piper_voice = _get_voice(voice)
        native_rate = piper_voice.config.sample_rate

        raw_parts = []
        for chunk in piper_voice.synthesize(text):
            raw_parts.append(chunk.audio_int16_bytes)

        if not raw_parts:
            return b""

        raw_pcm = b"".join(raw_parts)
        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float64)
        num_output = int(len(samples) * TARGET_RATE / native_rate)
        resampled = scipy_resample(samples, num_output)
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()

    def list_voices(self) -> list[str]:
        return [v["id"] for v in VOICE_CATALOG]
```

**Step 3: Run tests**

Run: `cd packages/engine-starter && python -m pytest tests/test_tts.py -v`
Expected: PASS (first run downloads ~30MB model).

**Step 4: Commit**

```bash
git add packages/engine-starter/engine_starter/tts.py packages/engine-starter/tests/test_tts.py
git commit -m "feat(engine-starter): add StarterTTS with Piper ONNX"
```

---

## Task 10: engine_starter.llm — StarterLLM (Ollama/Claude/OpenAI)

**Files:**
- Create: `packages/engine-starter/engine_starter/llm.py`
- Create: `packages/engine-starter/tests/test_llm.py`

**Source:** Extract from `iphone-and-companion-transcribe-mode/engine/llm.py`.

**Step 1: Write failing tests**

```python
# packages/engine-starter/tests/test_llm.py
"""Tests for engine_starter.llm — StarterLLM."""

import pytest

from engine_starter.interfaces import LLMProvider
from engine_starter.llm import StarterLLM, _resolve_provider


class TestStarterLLM:
    def test_is_llm_provider(self):
        assert issubclass(StarterLLM, LLMProvider)


class TestResolveProvider:
    def test_defaults_to_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _resolve_provider() == "ollama"

    def test_explicit_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "claude")
        assert _resolve_provider() == "claude"

    def test_auto_detect_claude(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _resolve_provider() == "claude"
```

**Step 2: Implement**

```python
# packages/engine-starter/engine_starter/llm.py
"""StarterLLM — multi-provider LLM wrapper (Claude, OpenAI, Ollama).

Auto-detects provider from environment: Claude > OpenAI > Ollama.
For production, implement LLMProvider directly with your preferred API.
"""

import asyncio
import functools
import json
import logging
import os
from typing import Optional

from engine_starter.interfaces import LLMProvider

log = logging.getLogger("engine_starter.llm")

CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
CLAUDE_SONNET = "claude-sonnet-4-6"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

_anthropic_client = None
_openai_client = None
_httpx_client = None


def _resolve_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider in ("claude", "openai", "ollama"):
        return provider
    if os.getenv("ANTHROPIC_API_KEY", ""):
        return "claude"
    if os.getenv("OPENAI_API_KEY", ""):
        return "openai"
    return "ollama"


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
        log.info("Anthropic client initialized")
    return _anthropic_client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
        log.info("OpenAI client initialized")
    return _openai_client


def _get_httpx():
    global _httpx_client
    if _httpx_client is None:
        import httpx
        _httpx_client = httpx.Client(timeout=120.0)
    return _httpx_client


def _generate_claude(system: str, messages: list[dict], model: str = "") -> str:
    client = _get_anthropic()
    active_model = model or CLAUDE_HAIKU
    resp = client.messages.create(
        model=active_model,
        max_tokens=1024 if "sonnet" in active_model else 300,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


def _generate_openai(system: str, messages: list[dict]) -> str:
    client = _get_openai()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_messages = [{"role": "system", "content": system}] + messages
    resp = client.chat.completions.create(model=model, max_tokens=300, messages=openai_messages)
    return resp.choices[0].message.content


def _generate_ollama(system: str, messages: list[dict], model: str = "") -> str:
    client = _get_httpx()
    active_model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
    ollama_messages = [{"role": "system", "content": system}] + messages
    resp = client.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": active_model, "messages": ollama_messages, "stream": False},
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _generate_sync(system: str, messages: list[dict], provider: str = "", model: str = "") -> str:
    provider = provider or _resolve_provider()
    if provider == "claude":
        return _generate_claude(system, messages, model=model)
    elif provider == "openai":
        return _generate_openai(system, messages)
    else:
        return _generate_ollama(system, messages, model=model)


class StarterLLM(LLMProvider):
    """Reference LLM using Ollama (local), Claude, or OpenAI.

    Auto-detects provider from env vars. Pass provider/model to override.
    """

    def __init__(self, provider: str = "", model: str = "", system_prompt: str = ""):
        self._provider = provider
        self._model = model
        self._system = system_prompt or "You are a helpful voice assistant. Keep responses concise and conversational."

    async def chat(self, messages: list[dict]) -> str:
        loop = asyncio.get_event_loop()
        fn = functools.partial(
            _generate_sync, self._system, messages, self._provider, self._model
        )
        return await loop.run_in_executor(None, fn)
```

**Step 3: Run tests**

Run: `cd packages/engine-starter && python -m pytest tests/test_llm.py -v`
Expected: All 4 tests PASS.

**Step 4: Commit**

```bash
git add packages/engine-starter/engine_starter/llm.py packages/engine-starter/tests/test_llm.py
git commit -m "feat(engine-starter): add StarterLLM with Claude/OpenAI/Ollama"
```

---

## Task 11: transport.tunnel — Cloudflare tunnel management

**Files:**
- Create: `packages/transport/transport/tunnel.py`
- Create: `packages/transport/tests/test_tunnel.py`

**Source:** Extract from `voice-calendar-scheduler-FSM/scripts/setup_tunnel.sh` and `scripts/run.sh`.

**Step 1: Write failing tests**

```python
# packages/transport/tests/test_tunnel.py
"""Tests for transport.tunnel — Cloudflare tunnel management."""

import pytest

from transport.tunnel import CloudflareTunnel


class TestCloudflareTunnel:
    def test_init_defaults(self):
        t = CloudflareTunnel(local_port=8090)
        assert t.local_port == 8090
        assert t.url is None

    def test_init_with_config(self, tmp_path):
        config = tmp_path / ".tunnel-config"
        config.write_text("TUNNEL_NAME=test-tunnel\nTUNNEL_ID=abc123\nTUNNEL_URL=https://test.example.com\n")
        t = CloudflareTunnel(local_port=8090, config_path=str(config))
        assert t._tunnel_name == "test-tunnel"
        assert t._tunnel_url == "https://test.example.com"

    def test_missing_config_uses_quick_tunnel(self, tmp_path):
        config = tmp_path / ".tunnel-config-missing"
        t = CloudflareTunnel(local_port=8090, config_path=str(config))
        assert t._tunnel_name == ""
```

**Step 2: Implement**

```python
# packages/transport/transport/tunnel.py
"""Cloudflare Tunnel management — named tunnels or quick (random URL) tunnels.

Wraps the `cloudflared` CLI binary as a subprocess.
"""

import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path

log = logging.getLogger("transport.tunnel")


class CloudflareTunnel:
    """Manage a Cloudflare tunnel for exposing a local server.

    Uses a named tunnel if .tunnel-config exists (from setup_tunnel.sh),
    otherwise falls back to a quick tunnel with a random *.trycloudflare.com URL.
    """

    def __init__(self, local_port: int, config_path: str = ".tunnel-config"):
        self.local_port = local_port
        self.url: str | None = None
        self._process: subprocess.Popen | None = None
        self._tunnel_name = ""
        self._tunnel_id = ""
        self._tunnel_url = ""

        config = Path(config_path)
        if config.exists():
            self._load_config(config)

    def _load_config(self, path: Path):
        """Parse KEY=VALUE config file."""
        for line in path.read_text().splitlines():
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key == "TUNNEL_NAME":
                self._tunnel_name = value
            elif key == "TUNNEL_ID":
                self._tunnel_id = value
            elif key == "TUNNEL_URL":
                self._tunnel_url = value

    def start(self):
        """Start the cloudflared tunnel subprocess."""
        if self._tunnel_name:
            # Named tunnel
            cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{self.local_port}", "run", self._tunnel_name]
            self.url = self._tunnel_url
            log.info("Starting named tunnel: %s → %s", self._tunnel_name, self.url)
        else:
            # Quick tunnel (random URL)
            cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{self.local_port}"]
            log.info("Starting quick tunnel on port %d", self.local_port)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # For quick tunnels, parse URL from cloudflared output
        if not self._tunnel_name:
            self._wait_for_url()

    def _wait_for_url(self, timeout: float = 30.0):
        """Parse the quick tunnel URL from cloudflared output."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._process is None or self._process.poll() is not None:
                break
            line = self._process.stdout.readline()
            if not line:
                continue
            match = re.search(r'(https://\S+\.trycloudflare\.com)', line)
            if match:
                self.url = match.group(1)
                log.info("Quick tunnel URL: %s", self.url)
                return
        log.warning("Could not determine tunnel URL within %.0fs", timeout)

    def stop(self):
        """Stop the cloudflared tunnel."""
        if self._process:
            log.info("Stopping tunnel")
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self.url = None
```

**Step 3: Run tests**

Run: `cd packages/transport && python -m pytest tests/test_tunnel.py -v`
Expected: All 3 tests PASS.

**Step 4: Commit**

```bash
git add packages/transport/transport/tunnel.py packages/transport/tests/test_tunnel.py
git commit -m "feat(transport): add CloudflareTunnel wrapper"
```

---

## Task 12: transport JS client — voice-webrtc-client.js

**Files:**
- Create: `packages/transport/js/voice-webrtc-client.js`

**Source:** Extract and refactor from `voice-calendar-scheduler-FSM/web/app.js`.

**Step 1: Write the JS client**

```javascript
// packages/transport/js/voice-webrtc-client.js
/**
 * VoiceWebRTCClient — framework-agnostic browser WebRTC client.
 *
 * Connects to a transport.signaling server, negotiates WebRTC,
 * and streams bidirectional audio.
 *
 * Usage:
 *   import { VoiceWebRTCClient } from 'voice-frontend/transport';
 *
 *   const client = new VoiceWebRTCClient({ signalingUrl: '/ws' });
 *   client.on('connected', () => console.log('Call active'));
 *   document.getElementById('call-btn').onclick = () => {
 *       if (client.inCall) client.hangUp();
 *       else client.startCall();
 *   };
 */

export class VoiceWebRTCClient {
    constructor(options = {}) {
        this.signalingUrl = options.signalingUrl || '/ws';
        this.audio = Object.assign({
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 48000,
        }, options.audio || {});
        this.iceGatheringTimeout = options.iceGatheringTimeout || 10000;

        this._ws = null;
        this._pc = null;
        this._localStream = null;
        this._iceServers = [];
        this._listeners = {};
        this.inCall = false;

        this._keepaliveId = null;
        this._connectSignaling();
    }

    // ── Event emitter ─────────────────────────────────────

    on(event, fn) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(fn);
    }

    _emit(event, ...args) {
        (this._listeners[event] || []).forEach(fn => fn(...args));
    }

    // ── Signaling ─────────────────────────────────────────

    _connectSignaling() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = this.signalingUrl.startsWith('ws')
            ? this.signalingUrl
            : proto + '//' + location.host + this.signalingUrl;

        this._emit('log', 'Connecting to ' + url, 'info');
        this._ws = new WebSocket(url);

        this._ws.onopen = () => {
            this._emit('log', 'Signaling connected', 'info');
            this._ws.send(JSON.stringify({ type: 'hello' }));
            this._startKeepalive();
        };

        this._ws.onmessage = (event) => {
            let msg;
            try { msg = JSON.parse(event.data); }
            catch { this._emit('log', 'Invalid JSON from server', 'error'); return; }

            switch (msg.type) {
                case 'hello_ack':
                    this._iceServers = msg.ice_servers || [];
                    this._emit('log', 'Got ' + this._iceServers.length + ' ICE server(s)', 'info');
                    this._emit('ready');
                    break;
                case 'webrtc_answer':
                    this._handleAnswer(msg.sdp);
                    break;
                case 'error':
                    this._emit('log', 'Server error: ' + msg.message, 'error');
                    if (this.inCall) this._cleanup();
                    break;
                case 'pong':
                    break;
                default:
                    this._emit('log', 'Unknown message: ' + msg.type, 'info');
            }
        };

        this._ws.onclose = () => {
            this._emit('log', 'Signaling disconnected', 'info');
            this._emit('ended');
            this._stopKeepalive();
            this._ws = null;
            if (this.inCall) this._cleanup();
        };

        this._ws.onerror = () => {
            this._emit('log', 'WebSocket error', 'error');
        };
    }

    _startKeepalive() {
        this._keepaliveId = setInterval(() => {
            if (this._ws && this._ws.readyState === WebSocket.OPEN) {
                this._ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    _stopKeepalive() {
        if (this._keepaliveId) {
            clearInterval(this._keepaliveId);
            this._keepaliveId = null;
        }
    }

    // ── WebRTC ────────────────────────────────────────────

    async startCall() {
        if (this.inCall) return;

        try {
            this._localStream = await navigator.mediaDevices.getUserMedia({
                audio: this.audio,
                video: false,
            });
        } catch (err) {
            this._emit('failed', 'Microphone access denied: ' + err.message);
            return;
        }

        const rtcConfig = {
            iceServers: this._iceServers.map(s => {
                const entry = { urls: s.urls || s.url || '' };
                if (s.username) entry.username = s.username;
                if (s.credential) entry.credential = s.credential;
                return entry;
            }),
        };

        this._pc = new RTCPeerConnection(rtcConfig);

        this._localStream.getTracks().forEach(t => this._pc.addTrack(t, this._localStream));

        this._pc.ontrack = (event) => {
            this._emit('log', 'Remote audio track received', 'info');
            const audio = document.createElement('audio');
            audio.autoplay = true;
            audio.playsInline = true;
            audio.srcObject = event.streams[0] || new MediaStream([event.track]);
            document.body.appendChild(audio);
        };

        this._pc.oniceconnectionstatechange = () => {
            const state = this._pc.iceConnectionState;
            this._emit('log', 'ICE state: ' + state, 'info');
            if (state === 'connected') this._emit('connected');
            else if (state === 'disconnected' || state === 'failed') {
                this._emit('failed', 'ICE ' + state);
                this.hangUp();
            }
        };

        try {
            const offer = await this._pc.createOffer();
            await this._pc.setLocalDescription(offer);

            // Wait for ICE gathering (required for TURN relay candidates)
            await new Promise((resolve) => {
                if (this._pc.iceGatheringState === 'complete') { resolve(); return; }
                const timer = setTimeout(() => {
                    this._emit('log', 'ICE gathering timed out, proceeding with partial candidates', 'error');
                    resolve();
                }, this.iceGatheringTimeout);
                this._pc.onicegatheringstatechange = () => {
                    if (this._pc.iceGatheringState === 'complete') {
                        clearTimeout(timer);
                        resolve();
                    }
                };
            });

            this._ws.send(JSON.stringify({
                type: 'webrtc_offer',
                sdp: this._pc.localDescription.sdp,
            }));

            this.inCall = true;
        } catch (err) {
            this._emit('failed', 'Offer creation failed: ' + err.message);
            this._cleanup();
        }
    }

    async _handleAnswer(sdp) {
        if (!this._pc) return;
        try {
            await this._pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp }));
        } catch (err) {
            this._emit('log', 'Failed to set answer: ' + err.message, 'error');
        }
    }

    hangUp() {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            this._ws.send(JSON.stringify({ type: 'hangup' }));
        }
        this._cleanup();
        this._emit('ended');
    }

    _cleanup() {
        this.inCall = false;
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._localStream) {
            this._localStream.getTracks().forEach(t => t.stop());
            this._localStream = null;
        }
    }

    destroy() {
        this.hangUp();
        this._stopKeepalive();
        if (this._ws) { this._ws.close(); this._ws = null; }
    }
}
```

**Step 2: Commit**

```bash
git add packages/transport/js/voice-webrtc-client.js
git commit -m "feat(transport): add VoiceWebRTCClient browser JS client"
```

---

## Task 13: edge_auth — Auth provider ABC, implementations, middleware

**Files:**
- Create: `packages/edge-auth/edge_auth/base.py`
- Create: `packages/edge-auth/edge_auth/providers/cloudflare.py`
- Create: `packages/edge-auth/edge_auth/providers/google.py`
- Create: `packages/edge-auth/edge_auth/providers/bearer.py`
- Create: `packages/edge-auth/edge_auth/providers/none.py`
- Create: `packages/edge-auth/edge_auth/middleware.py`
- Create: `packages/edge-auth/edge_auth/composite.py`
- Create: `packages/edge-auth/tests/test_auth.py`

**Source:** Extract from `iphone-and-companion-transcribe-mode/gateway/auth.py` and `voice-calendar-scheduler-FSM/scheduling/auth.py`.

**Step 1: Write failing tests**

```python
# packages/edge-auth/tests/test_auth.py
"""Tests for edge_auth — providers, middleware, composite."""

import pytest

from edge_auth.base import AuthProvider, AuthResult
from edge_auth.providers.bearer import BearerTokenProvider
from edge_auth.providers.none import NoAuthProvider
from edge_auth.composite import CompositeProvider


class FakeRequest:
    def __init__(self, headers=None, query_params=None):
        self.headers = headers or {}
        self.query_params = query_params or {}


class FakeWebSocket:
    def __init__(self, query_params=None):
        self.query_params = query_params or {}


class TestAuthResult:
    def test_defaults(self):
        r = AuthResult(authenticated=True)
        assert r.user_id is None
        assert r.provider == ""

    def test_with_fields(self):
        r = AuthResult(authenticated=True, user_id="123", provider="bearer")
        assert r.user_id == "123"


class TestNoAuthProvider:
    @pytest.mark.asyncio
    async def test_always_authenticates(self):
        auth = NoAuthProvider()
        result = await auth.authenticate(FakeRequest())
        assert result.authenticated is True
        assert result.provider == "none"

    @pytest.mark.asyncio
    async def test_ws_always_authenticates(self):
        auth = NoAuthProvider()
        result = await auth.authenticate_ws(FakeWebSocket())
        assert result.authenticated is True


class TestBearerTokenProvider:
    @pytest.mark.asyncio
    async def test_valid_token(self):
        auth = BearerTokenProvider(token="secret123")
        req = FakeRequest(headers={"authorization": "Bearer secret123"})
        result = await auth.authenticate(req)
        assert result.authenticated is True

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        auth = BearerTokenProvider(token="secret123")
        req = FakeRequest(headers={"authorization": "Bearer wrong"})
        result = await auth.authenticate(req)
        assert result.authenticated is False

    @pytest.mark.asyncio
    async def test_missing_header(self):
        auth = BearerTokenProvider(token="secret123")
        req = FakeRequest()
        result = await auth.authenticate(req)
        assert result.authenticated is False

    @pytest.mark.asyncio
    async def test_ws_via_query_param(self):
        auth = BearerTokenProvider(token="secret123")
        ws = FakeWebSocket(query_params={"token": "secret123"})
        result = await auth.authenticate_ws(ws)
        assert result.authenticated is True


class TestCompositeProvider:
    @pytest.mark.asyncio
    async def test_first_match_wins(self):
        auth = CompositeProvider([
            BearerTokenProvider(token="key1"),
            NoAuthProvider(),
        ])
        req = FakeRequest(headers={"authorization": "Bearer wrong"})
        result = await auth.authenticate(req)
        # Bearer fails, NoAuth succeeds
        assert result.authenticated is True
        assert result.provider == "none"

    @pytest.mark.asyncio
    async def test_skips_none_providers(self):
        auth = CompositeProvider([None, NoAuthProvider()])
        result = await auth.authenticate(FakeRequest())
        assert result.authenticated is True
```

**Step 2: Implement all auth files**

`packages/edge-auth/edge_auth/base.py`:
```python
"""Auth provider base class and result type."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AuthResult:
    """Result of an authentication attempt."""
    authenticated: bool
    user_id: str | None = None
    user_email: str | None = None
    user_name: str | None = None
    provider: str = ""


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    async def authenticate(self, request) -> AuthResult:
        """Validate an HTTP request. Return AuthResult."""
        ...

    async def authenticate_ws(self, websocket) -> AuthResult:
        """Validate a WebSocket connection.

        Default: check ?token= query param. Override for custom behavior.
        """
        return await self.authenticate(websocket)
```

`packages/edge-auth/edge_auth/providers/none.py`:
```python
"""NoAuthProvider — allows everything. For local development only."""

from edge_auth.base import AuthProvider, AuthResult


class NoAuthProvider(AuthProvider):
    async def authenticate(self, request) -> AuthResult:
        return AuthResult(authenticated=True, provider="none")

    async def authenticate_ws(self, websocket) -> AuthResult:
        return AuthResult(authenticated=True, provider="none")
```

`packages/edge-auth/edge_auth/providers/bearer.py`:
```python
"""BearerTokenProvider — simple API key in Authorization header."""

from edge_auth.base import AuthProvider, AuthResult


class BearerTokenProvider(AuthProvider):
    def __init__(self, token: str):
        self._token = token

    async def authenticate(self, request) -> AuthResult:
        auth_header = getattr(request, 'headers', {}).get("authorization", "")
        if auth_header.startswith("Bearer ") and auth_header[7:] == self._token:
            return AuthResult(authenticated=True, provider="bearer")
        return AuthResult(authenticated=False, provider="bearer")

    async def authenticate_ws(self, websocket) -> AuthResult:
        token = getattr(websocket, 'query_params', {}).get("token", "")
        if token == self._token:
            return AuthResult(authenticated=True, provider="bearer")
        return AuthResult(authenticated=False, provider="bearer")
```

`packages/edge-auth/edge_auth/providers/cloudflare.py`:
```python
"""CloudflareAccessProvider — validates CF-Access-JWT-Assertion header."""

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

            # Fetch Cloudflare's public keys
            async with httpx.AsyncClient() as client:
                resp = await client.get(self._certs_url)
                resp.raise_for_status()
                jwks = resp.json()

            # Decode and verify the JWT
            public_keys = jwt.PyJWKSet.from_dict(jwks)
            header = jwt.get_unverified_header(token)
            key = public_keys[header["kid"]]

            payload = jwt.decode(
                token,
                key=key.key,
                algorithms=["RS256"],
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
```

`packages/edge-auth/edge_auth/providers/google.py`:
```python
"""GoogleJWTProvider — validates Google OAuth2 ID tokens."""

import logging
import os

from edge_auth.base import AuthProvider, AuthResult

log = logging.getLogger("edge_auth.google")


class GoogleJWTProvider(AuthProvider):
    """Validate Google Sign-In JWT tokens.

    Requires: pip install voice-frontend-edge-auth[google]
    """

    def __init__(self, client_id: str = "", allowed_emails: set[str] | None = None):
        self._client_id = client_id or os.getenv("GOOGLE_CLIENT_ID", "")
        self._allowed_emails = allowed_emails

    async def authenticate(self, request) -> AuthResult:
        auth_header = getattr(request, 'headers', {}).get("authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        if not token:
            return AuthResult(authenticated=False, provider="google")
        return self._verify(token)

    async def authenticate_ws(self, websocket) -> AuthResult:
        token = getattr(websocket, 'query_params', {}).get("google_jwt", "")
        if not token:
            return AuthResult(authenticated=False, provider="google")
        return self._verify(token)

    def _verify(self, token: str) -> AuthResult:
        try:
            from google.auth.transport import requests as google_requests
            from google.oauth2 import id_token

            idinfo = id_token.verify_oauth2_token(
                token, google_requests.Request(), self._client_id
            )

            if idinfo.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
                return AuthResult(authenticated=False, provider="google")

            if not idinfo.get("email_verified"):
                return AuthResult(authenticated=False, provider="google")

            email = idinfo.get("email", "").lower()
            if self._allowed_emails and email not in self._allowed_emails:
                return AuthResult(authenticated=False, provider="google")

            return AuthResult(
                authenticated=True,
                user_id=idinfo.get("sub"),
                user_email=email,
                user_name=idinfo.get("name"),
                provider="google",
            )
        except Exception as e:
            log.warning("Google auth failed: %s", e)
            return AuthResult(authenticated=False, provider="google")
```

`packages/edge-auth/edge_auth/composite.py`:
```python
"""CompositeProvider — try multiple auth providers in order."""

from edge_auth.base import AuthProvider, AuthResult


class CompositeProvider(AuthProvider):
    """Try multiple providers in order. First successful auth wins."""

    def __init__(self, providers: list[AuthProvider | None]):
        self._providers = [p for p in providers if p is not None]

    async def authenticate(self, request) -> AuthResult:
        for provider in self._providers:
            result = await provider.authenticate(request)
            if result.authenticated:
                return result
        return AuthResult(authenticated=False)

    async def authenticate_ws(self, websocket) -> AuthResult:
        for provider in self._providers:
            result = await provider.authenticate_ws(websocket)
            if result.authenticated:
                return result
        return AuthResult(authenticated=False)
```

`packages/edge-auth/edge_auth/middleware.py`:
```python
"""FastAPI middleware and dependency for auth."""

from typing import Callable

from edge_auth.base import AuthProvider, AuthResult


def auth_middleware(provider: AuthProvider, exclude: list[str] | None = None):
    """Create a Starlette middleware class that authenticates all requests.

    Args:
        provider: The auth provider to use.
        exclude: List of path prefixes to skip auth for (e.g., ["/health", "/ws"]).
    """
    exclude = exclude or []

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class _AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            for prefix in exclude:
                if request.url.path.startswith(prefix):
                    return await call_next(request)

            result = await provider.authenticate(request)
            if not result.authenticated:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

            request.state.auth = result
            return await call_next(request)

    return _AuthMiddleware


def auth_dependency(provider: AuthProvider) -> Callable:
    """Create a FastAPI dependency that authenticates the request.

    Usage:
        require_auth = auth_dependency(auth)

        @app.get("/api/config", dependencies=[Depends(require_auth)])
        async def get_config(): ...
    """
    from fastapi import HTTPException, Request

    async def _dependency(request: Request) -> AuthResult:
        result = await provider.authenticate(request)
        if not result.authenticated:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return result

    return _dependency
```

**Step 3: Run tests**

Run: `cd packages/edge-auth && python -m pytest tests/test_auth.py -v`
Expected: All 10 tests PASS.

**Step 4: Commit**

```bash
git add packages/edge-auth/
git commit -m "feat(edge-auth): add auth providers (CF Access, Google JWT, Bearer, NoAuth), middleware, composite"
```

---

## Task 14: Examples — minimal-voice-app, with-auth, custom-engine

**Files:**
- Create: `examples/minimal-voice-app/server.py`
- Create: `examples/minimal-voice-app/index.html`
- Create: `examples/with-auth/server.py`
- Create: `examples/custom-engine/server.py`

**Step 1: Create minimal-voice-app**

`examples/minimal-voice-app/server.py`:
```python
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
```

`examples/minimal-voice-app/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Minimal Voice App</title>
    <style>
        body { font-family: system-ui; max-width: 400px; margin: 40px auto; text-align: center; }
        button { padding: 16px 32px; font-size: 18px; cursor: pointer; border-radius: 8px; border: none; background: #4CAF50; color: white; }
        button:disabled { opacity: 0.5; }
        button.hangup { background: #f44336; }
        #status { margin: 20px 0; color: #666; }
    </style>
</head>
<body>
    <h1>Voice App</h1>
    <p id="status">Connecting...</p>
    <button id="call-btn" disabled>Call</button>
    <script type="module">
        import { VoiceWebRTCClient } from './voice-webrtc-client.js';

        const client = new VoiceWebRTCClient({ signalingUrl: '/ws' });
        const btn = document.getElementById('call-btn');
        const status = document.getElementById('status');

        client.on('ready', () => { status.textContent = 'Ready'; btn.disabled = false; });
        client.on('connected', () => { status.textContent = 'In call'; });
        client.on('ended', () => { status.textContent = 'Ready'; btn.textContent = 'Call'; btn.classList.remove('hangup'); });
        client.on('failed', (r) => { status.textContent = 'Error: ' + r; });

        btn.onclick = () => {
            if (client.inCall) { client.hangUp(); }
            else { client.startCall(); btn.textContent = 'Hang Up'; btn.classList.add('hangup'); }
        };
    </script>
</body>
</html>
```

`examples/with-auth/server.py`:
```python
"""Voice app with Cloudflare Access authentication.

Run: uvicorn server:app --port 8090
"""

from fastapi import FastAPI, WebSocket, Depends

from transport.turn import TwilioTURN
from transport.signaling import SignalingServer
from transport.session import WebRTCSession
from edge_auth import auth_middleware, auth_dependency, CompositeProvider
from edge_auth.providers import CloudflareAccessProvider, NoAuthProvider
from engine_starter.stt import StarterSTT
from engine_starter.tts import StarterTTS
from engine_starter.llm import StarterLLM

import os

DEBUG = os.getenv("DEBUG", "").lower() == "true"

app = FastAPI()
stt, tts, llm = StarterSTT(), StarterTTS(), StarterLLM()

auth = CompositeProvider([
    CloudflareAccessProvider(team_domain="myteam.cloudflareaccess.com"),
    NoAuthProvider() if DEBUG else None,
])

# Protect all HTTP endpoints except health and WS
app.add_middleware(auth_middleware(auth, exclude=["/health", "/ws"]))
require_auth = auth_dependency(auth)


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
```

`examples/custom-engine/server.py`:
```python
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


# ── Custom TTS provider ───────────────────────────────────
# Replace this with your production TTS (ElevenLabs, Azure, etc.)

class MyProductionTTS(TTSProvider):
    """Example: implement TTSProvider with your preferred TTS service."""

    def synthesize(self, text: str, voice: str = "") -> bytes:
        # YOUR CODE HERE: call your TTS API, return 48kHz int16 PCM bytes
        # Example with ElevenLabs:
        #   response = elevenlabs.generate(text=text, voice=voice)
        #   return resample_to_48k(response.audio_bytes)
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
```

**Step 2: Commit**

```bash
git add examples/
git commit -m "feat: add examples — minimal-voice-app, with-auth, custom-engine"
```

---

## Task 15: Integration test and final wiring

**Files:**
- Create: `packages/transport/tests/conftest.py`
- Create: `packages/transport/tests/test_integration.py`
- Create: `pytest.ini` (root)

**Step 1: Create root pytest.ini**

```ini
[pytest]
testpaths = packages/transport/tests packages/edge-auth/tests packages/engine-starter/tests
asyncio_mode = auto
```

**Step 2: Write integration test**

```python
# packages/transport/tests/test_integration.py
"""Integration test — verify the full signaling + session + TTS pipeline compiles."""

import json
import pytest

from transport.turn import StaticICE
from transport.signaling import SignalingServer
from transport.audio import AudioQueue, resample


class TestFullPipelineImports:
    """Verify all public APIs can be imported."""

    def test_transport_imports(self):
        from transport.turn import TURNProvider, TwilioTURN, StaticICE
        from transport.signaling import SignalingServer
        from transport.session import WebRTCSession
        from transport.tunnel import CloudflareTunnel
        from transport.audio import AudioQueue, resample

    def test_edge_auth_imports(self):
        from edge_auth import AuthProvider, AuthResult, auth_middleware, auth_dependency, CompositeProvider
        from edge_auth.providers import CloudflareAccessProvider, GoogleJWTProvider, BearerTokenProvider, NoAuthProvider

    def test_engine_starter_imports(self):
        from engine_starter import STTProvider, TTSProvider, LLMProvider, TranscriptionResult
        from engine_starter.interfaces import AudioChunk


class TestResampleRoundTrip:
    def test_48k_to_16k_and_back(self):
        import numpy as np
        original = np.random.randint(-1000, 1000, size=960, dtype=np.int16).tobytes()
        down = resample(original, from_rate=48000, to_rate=16000)
        up = resample(down, from_rate=16000, to_rate=48000)
        # Should be same length (may differ slightly in values due to interpolation)
        assert len(np.frombuffer(up, dtype=np.int16)) == 960
```

**Step 3: Run all tests**

Run: `python -m pytest -v`
Expected: All tests PASS.

**Step 4: Commit**

```bash
git add pytest.ini packages/transport/tests/test_integration.py
git commit -m "test: add integration tests and root pytest config"
```

---

## Task 16: Update transport __init__.py with public API exports

**Files:**
- Modify: `packages/transport/transport/__init__.py`

**Step 1: Update**

```python
# packages/transport/transport/__init__.py
"""Voice Frontend Transport — WebRTC connectivity for voice applications."""

from transport.turn import TURNProvider, TwilioTURN, StaticICE
from transport.signaling import SignalingServer
from transport.session import WebRTCSession
from transport.tunnel import CloudflareTunnel
from transport.audio import AudioQueue, resample

__all__ = [
    "TURNProvider",
    "TwilioTURN",
    "StaticICE",
    "SignalingServer",
    "WebRTCSession",
    "CloudflareTunnel",
    "AudioQueue",
    "resample",
]
```

**Step 2: Run all tests one final time**

Run: `python -m pytest -v`
Expected: All tests PASS.

**Step 3: Commit**

```bash
git add packages/transport/transport/__init__.py
git commit -m "feat(transport): export public API from __init__.py"
```

---

## Summary

| Task | Package | What | Test count |
|------|---------|------|------------|
| 1 | all | pyproject.toml + __init__.py scaffolding | 0 |
| 2 | transport | AudioQueue + resample | 10 |
| 3 | transport | TURN providers (ABC + Twilio + Static) | 4 |
| 4 | transport | WebRTCAudioSource (aiortc track) | 3 |
| 5 | transport | WebRTCSession (speak/listen/VAD) | 7 |
| 6 | transport | SignalingServer (WebSocket) | 3 |
| 7 | engine-starter | ABCs + data classes | 5 |
| 8 | engine-starter | StarterSTT (faster-whisper) | 3 |
| 9 | engine-starter | StarterTTS (Piper) | 5 |
| 10 | engine-starter | StarterLLM (Claude/OpenAI/Ollama) | 4 |
| 11 | transport | CloudflareTunnel | 3 |
| 12 | transport | voice-webrtc-client.js | 0 |
| 13 | edge-auth | All auth providers + middleware + composite | 10 |
| 14 | all | Examples (minimal, with-auth, custom-engine) | 0 |
| 15 | all | Integration tests + pytest config | 3 |
| 16 | transport | Final public API exports | 0 |
| **Total** | | | **~60** |

**Dependency order:** Task 7 (interfaces/ABCs) should run before Tasks 4-5 (they reference AudioChunk). All other tasks are largely independent and can be parallelized.
