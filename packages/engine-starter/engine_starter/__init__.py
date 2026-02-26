"""Voice Frontend Engine Starter — reference STT/TTS/LLM implementations.

WARNING: This is a starter engine for testing, not production.
Swap providers via the ABCs in engine_starter.interfaces.
"""

from engine_starter.interfaces import STTProvider, TTSProvider, LLMProvider, TranscriptionResult
from engine_starter.config import EngineConfig
from engine_starter.kokoro_tts import KokoroTTS
from engine_starter.admin import create_admin_router

__all__ = [
    "STTProvider",
    "TTSProvider",
    "LLMProvider",
    "TranscriptionResult",
    "EngineConfig",
    "KokoroTTS",
    "create_admin_router",
]
