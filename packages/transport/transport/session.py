"""WebRTC session — one active voice call.

Manages the aiortc RTCPeerConnection, mic audio capture, TTS playback,
and provides high-level speak()/listen() APIs.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

from transport.audio import AudioQueue
from transport.audio_source import WebRTCAudioSource, FRAME_SAMPLES, SAMPLE_RATE


@dataclass
class AudioChunk:
    """A chunk of PCM audio data. Defined here to avoid cross-package imports."""
    samples: bytes
    sample_rate: int
    channels: int

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
        self.vad_silence_gap: int = 15
        self.vad_speech_confirm_frames: int = 2

        # Barge-in settings (live-tunable)
        self.barge_in_enabled: bool = True
        self.barge_in_energy_threshold: int = 600
        self.barge_in_confirm_frames: int = 2

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
            self._mic_recv_task = asyncio.create_task(self._recv_mic_audio(track))

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
            tts: Any object with a synthesize(text, voice) method (sync or async).
        """
        self._audio_source.set_generator(self._tts_generator)
        text = self._clean_for_speech(text)
        sentences = self._split_sentences(text)
        if not sentences:
            return 0.0

        log.info("TTS: %d sentences to synthesize", len(sentences))
        loop = asyncio.get_running_loop()
        total_bytes = 0

        for sentence in sentences:
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

    @staticmethod
    def _compute_rms(pcm_bytes: bytes) -> float:
        """Compute RMS energy of int16 PCM audio."""
        if not pcm_bytes:
            return 0.0
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    async def speak_with_barge_in(
        self, text: str, tts, stt=None
    ) -> tuple[float, str | None]:
        """Speak with barge-in detection — stops playback if user interrupts.

        Returns (duration_played, transcribed_text | None).
        If barge-in is disabled, behaves like regular speak().
        """
        if not self.barge_in_enabled:
            dur = await self.speak(text, tts)
            return dur, None

        # Enqueue all TTS audio (reuses speak() logic)
        self._audio_source.set_generator(self._tts_generator)
        cleaned = self._clean_for_speech(text)
        sentences = self._split_sentences(cleaned)
        if not sentences:
            return 0.0, None

        loop = asyncio.get_running_loop()
        total_bytes = 0
        for sentence in sentences:
            if asyncio.iscoroutinefunction(getattr(tts, 'synthesize', None)):
                pcm = await tts.synthesize(sentence)
            else:
                pcm = await loop.run_in_executor(None, tts.synthesize, sentence)
            if pcm:
                self._audio_queue.enqueue(pcm)
                total_bytes += len(pcm)

        if total_bytes == 0:
            return 0.0, None

        total_duration = total_bytes / (SAMPLE_RATE * 2)
        log.info("Barge-in TTS: %d bytes, %.1fs playback", total_bytes, total_duration)

        # Enable recording and clear mic buffer for fresh barge-in detection
        self._mic_frames.clear()
        self._recording = True
        consecutive_speech = 0
        interrupted = False
        elapsed = 0.0
        poll_interval = 0.1

        while self._audio_queue.is_playing:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            # Check recent mic frames for speech energy
            recent = self._mic_frames[-5:] if len(self._mic_frames) >= 5 else self._mic_frames
            if recent:
                pcm_data = b"".join(recent)
                rms = self._compute_rms(pcm_data)
                if rms >= self.barge_in_energy_threshold:
                    consecutive_speech += 1
                    if consecutive_speech >= self.barge_in_confirm_frames:
                        log.info("Barge-in detected at %.1fs (RMS=%.0f)", elapsed, rms)
                        self.stop_speaking()
                        interrupted = True
                        break
                else:
                    consecutive_speech = 0

        self._recording = False

        if interrupted and stt and self._mic_frames:
            captured = b"".join(self._mic_frames)
            result = await loop.run_in_executor(
                None, stt.transcribe, captured, SAMPLE_RATE
            )
            transcription = result.text if hasattr(result, 'text') else result[0]
            return elapsed, transcription.strip() if transcription else None

        return (elapsed if interrupted else total_duration), None

    async def listen(self, stt) -> AsyncIterator[str]:
        """Yield transcribed utterances using energy-based VAD + STT.

        Args:
            stt: Any object with a transcribe(audio_bytes, sample_rate) method.
        """
        self._mic_frames.clear()
        self._recording = True
        log.info("Listening started")

        for _ in range(50):
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
        loop = asyncio.get_running_loop()

        try:
            while self._recording:
                await asyncio.sleep(0.1)
                if not self._mic_frames:
                    continue

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
