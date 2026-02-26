"""KokoroTTS — Kokoro ONNX text-to-speech provider.

Lazy-loads kokoro_onnx from KOKORO_MODELS_DIR on first synthesis.
Outputs 48kHz mono int16 PCM, resampled from Kokoro's native rate
using scipy's polyphase filter (matching StarterTTS quality).
"""

import logging
import os
import threading
from pathlib import Path

import numpy as np
from scipy.signal import resample as scipy_resample

from engine_starter.interfaces import TTSProvider

log = logging.getLogger("engine_starter.kokoro_tts")

TARGET_RATE = 48000

_default_model_dir = Path.home() / ".cache" / "voice-frontend" / "kokoro"
MODEL_DIR = Path(os.environ.get("KOKORO_MODELS_DIR", _default_model_dir))

# Voice prefix -> BCP-47 language code for Kokoro synthesis.
LANG_MAP = {
    "a": "en-us",
    "b": "en-gb",
    "f": "fr-fr",
    "i": "it",
    "j": "ja",
    "z": "cmn",
}

# Voice prefix -> human-readable group label for admin UI grouping.
VOICE_GROUPS = {
    "af_": "US English - Female",
    "am_": "US English - Male",
    "bf_": "British English - Female",
    "bm_": "British English - Male",
    "ff_": "French - Female",
    "if_": "Italian - Female",
    "jf_": "Japanese - Female",
    "zf_": "Chinese - Female",
}

_kokoro_instance = None
_kokoro_lock = threading.Lock()


def _get_kokoro():
    """Lazy-load and cache the Kokoro ONNX model (thread-safe)."""
    global _kokoro_instance
    if _kokoro_instance is not None:
        return _kokoro_instance

    with _kokoro_lock:
        # Double-checked locking — another thread may have loaded while we waited
        if _kokoro_instance is not None:
            return _kokoro_instance

        from kokoro_onnx import Kokoro

        voices_path = MODEL_DIR / "voices.bin"
        model_path = MODEL_DIR / "kokoro-v1.0.onnx"

        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found in {MODEL_DIR}. "
                f"Expected: {model_path.name} and {voices_path.name}. "
                f"Download from https://github.com/thewh1teagle/kokoro-onnx/releases "
                f"or set KOKORO_MODELS_DIR to the correct path."
            )

        log.info("Loading Kokoro model from %s ...", MODEL_DIR)
        _kokoro_instance = Kokoro(str(model_path), str(voices_path))
        log.info("Kokoro model loaded")
        return _kokoro_instance


class KokoroTTS(TTSProvider):
    """TTS provider using Kokoro ONNX. Lazy-loads model on first use.

    Outputs 48kHz mono int16 PCM suitable for WebRTC playback.
    Voice IDs follow Kokoro's naming convention (e.g. ``af_heart``).
    The first character selects the language via LANG_MAP.
    """

    DEFAULT_VOICE = "af_heart"

    def synthesize(self, text: str, voice: str = "") -> bytes:
        """Convert text to 48kHz int16 PCM bytes.

        Args:
            text: Text to synthesize. Empty/whitespace returns b"".
            voice: Kokoro voice ID (e.g. "af_heart"). Defaults to DEFAULT_VOICE.
        """
        if not text or not text.strip():
            return b""

        kokoro = _get_kokoro()
        voice = voice or self.DEFAULT_VOICE
        lang = LANG_MAP.get(voice[0], "en-us")

        samples, native_rate = kokoro.create(text, voice=voice, lang=lang)

        if len(samples) == 0:
            return b""

        # Resample to 48kHz using polyphase filter (same quality as StarterTTS)
        if native_rate != TARGET_RATE:
            num_output = int(len(samples) * TARGET_RATE / native_rate)
            resampled = scipy_resample(samples.astype(np.float64), num_output)
        else:
            resampled = samples.astype(np.float64)

        pcm = np.clip(resampled * 32767, -32768, 32767).astype(np.int16)
        return pcm.tobytes()

    def list_voices(self) -> list[str]:
        """Return sorted list of available Kokoro voice IDs."""
        kokoro = _get_kokoro()
        return sorted(kokoro.get_voices())
