"""Provider interfaces — ABCs for STT, TTS, and LLM.

These are the contracts that production providers must implement.
The starter implementations in this package satisfy them for testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
    def transcribe(
        self, audio: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
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
