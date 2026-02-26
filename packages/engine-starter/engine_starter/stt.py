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

    log.info("Loading faster-whisper model: tiny (first run downloads ~39MB)...")
    _model = WhisperModel("tiny", device="cpu", compute_type="int8")
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

        segments, info = model.transcribe(samples, beam_size=1, language="en", vad_filter=True)

        text_parts = []
        worst_no_speech = 0.0
        for segment in segments:
            text_parts.append(segment.text.strip())
            worst_no_speech = max(worst_no_speech, segment.no_speech_prob)

        text = " ".join(text_parts).strip()
        log.info("Transcription: %r (no_speech=%.2f)", text[:100], worst_no_speech)
        return TranscriptionResult(text=text, no_speech_probability=worst_no_speech)
