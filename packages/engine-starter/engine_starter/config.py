"""Runtime configuration store — thread-safe, in-memory, known-keys-only.

Used by admin API and voice loop to share live-tunable settings.
Follows the AudioQueue._lock pattern from transport/audio.py.

Usage::

    from engine_starter.config import EngineConfig

    config = EngineConfig()
    config["vad_energy_threshold"] = 800  # set a value
    snap = config.snapshot()              # get full config as plain dict
    config.update({"tts_voice": "alba"})  # partial update
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
    """Thread-safe in-memory config store. Only allows known keys.

    All reads and writes are guarded by a lock, making it safe to read
    from the audio loop while writing from the admin API.
    """

    KNOWN_KEYS = frozenset(_DEFAULTS)

    def __init__(self):
        self._data = dict(_DEFAULTS)
        self._lock = threading.Lock()

    def __getitem__(self, key: str):
        """Get value by key. Raises KeyError for unknown keys."""
        with self._lock:
            if key not in _DEFAULTS:
                raise KeyError(f"Unknown config key: {key!r}")
            return self._data[key]

    def __setitem__(self, key: str, value):
        """Set value by key. Raises KeyError for unknown keys."""
        with self._lock:
            if key not in _DEFAULTS:
                raise KeyError(f"Unknown config key: {key!r}")
            self._data[key] = value

    def get(self, key: str, default=None):
        """Get value by key, returning *default* for unknown keys.

        Unlike ``__getitem__``, this does NOT raise for unknown keys —
        useful when reading optional keys that may not exist in older
        config versions.
        """
        with self._lock:
            if key not in _DEFAULTS:
                return default
            return self._data[key]

    def update(self, patch: dict) -> dict:
        """Apply partial update. Raises KeyError for unknown keys.

        Returns:
            Full config snapshot after applying the update.
        """
        with self._lock:
            for key in patch:
                if key not in _DEFAULTS:
                    raise KeyError(f"Unknown config key: {key!r}")
            self._data.update(patch)
            return dict(self._data)

    def snapshot(self) -> dict:
        """Return a shallow copy of the current config as a plain dict."""
        with self._lock:
            return dict(self._data)
