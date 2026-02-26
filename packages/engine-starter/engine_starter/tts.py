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
