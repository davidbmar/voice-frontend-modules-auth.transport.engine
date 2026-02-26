"""Runtime configuration store — thread-safe, in-memory, known-keys-only.

Used by admin API and voice loop to share live-tunable settings.
Follows the AudioQueue._lock pattern from transport/audio.py.
"""

import threading


_DEFAULTS = {
    "tts_engine": "piper",
    "tts_voice": "",
    "vad_energy_threshold": 500,
    "vad_speech_confirm_frames": 2,
    "vad_silence_gap": 15,
    "barge_in_enabled": True,
    "barge_in_energy_threshold": 600,
    "barge_in_confirm_frames": 2,
}


class EngineConfig:
    """Thread-safe in-memory config store. Only allows known keys."""

    def __init__(self):
        self._data = dict(_DEFAULTS)
        self._lock = threading.Lock()

    def __getitem__(self, key: str):
        with self._lock:
            if key not in _DEFAULTS:
                raise KeyError(f"Unknown config key: {key!r}")
            return self._data[key]

    def __setitem__(self, key: str, value):
        with self._lock:
            if key not in _DEFAULTS:
                raise KeyError(f"Unknown config key: {key!r}")
            self._data[key] = value

    def get(self, key: str, default=None):
        with self._lock:
            if key not in _DEFAULTS:
                return default
            return self._data[key]

    def update(self, patch: dict) -> dict:
        """Apply partial update. Returns full config after update."""
        with self._lock:
            for key in patch:
                if key not in _DEFAULTS:
                    raise KeyError(f"Unknown config key: {key!r}")
            self._data.update(patch)
            return dict(self._data)

    def snapshot(self) -> dict:
        """Return a copy of the current config."""
        with self._lock:
            return dict(self._data)
