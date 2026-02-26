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
