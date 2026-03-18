"""
STT capability manager — loads/unloads engines, dispatches transcription.
Supports Qwen3-ASR variants (mlx_audio) and Kotoba-Whisper (mlx_whisper).
The "remote" key is a sentinel handled directly by main.py's /stt endpoint.
"""
import threading
import time
from typing import Any

from modules.stt.engines import stt_qwen3, stt_whisper, stt_remote

# ── Engine registry ───────────────────────────────────────────────────────────
# backend: string key used by /stt/models API response
# engine:  module that implements load() / transcribe()

MODELS: dict[str, dict] = {
    "qwen3-0.6b-4bit": {"repo": "mlx-community/Qwen3-ASR-0.6B-4bit",  "backend": "mlx_audio",   "engine": stt_qwen3,  "label": "Qwen3 0.6B 4bit"},
    "qwen3-0.6b-8bit": {"repo": "mlx-community/Qwen3-ASR-0.6B-8bit",  "backend": "mlx_audio",   "engine": stt_qwen3,  "label": "Qwen3 0.6B 8bit"},
    "qwen3-0.6b-bf16": {"repo": "mlx-community/Qwen3-ASR-0.6B-bf16",  "backend": "mlx_audio",   "engine": stt_qwen3,  "label": "Qwen3 0.6B bf16"},
    "qwen3-1.7b-4bit": {"repo": "mlx-community/Qwen3-ASR-1.7B-4bit",  "backend": "mlx_audio",   "engine": stt_qwen3,  "label": "Qwen3 1.7B 4bit"},
    "qwen3-1.7b-8bit": {"repo": "mlx-community/Qwen3-ASR-1.7B-8bit",  "backend": "mlx_audio",   "engine": stt_qwen3,  "label": "Qwen3 1.7B 8bit"},
    "qwen3-1.7b-bf16": {"repo": "mlx-community/Qwen3-ASR-1.7B-bf16",  "backend": "mlx_audio",   "engine": stt_qwen3,  "label": "Qwen3 1.7B bf16"},
    "kotoba-whisper":  {"repo": "kotoba-tech/kotoba-whisper-v2.0",      "backend": "mlx_whisper", "engine": stt_whisper, "label": "Kotoba Whisper v2"},
    "remote":          {"repo": None,                                    "backend": "remote",      "engine": stt_remote,  "label": "Remote (OpenAI-compat)"},
}

DEFAULT_MODEL = "qwen3-0.6b-4bit"

# Available backends — checked at import time via engine modules
AVAILABLE: dict[str, bool] = {
    "mlx_audio":   stt_qwen3.AVAILABLE,
    "mlx_whisper": stt_whisper.AVAILABLE,
    "remote":      True,
}


class STTManager:
    def __init__(self):
        self.active_key: str = DEFAULT_MODEL
        self.loaded_model: Any = None
        self._cancel_flag = threading.Event()
        self._load_lock = threading.Lock()

    def transcribe(self, wav_path: str, language: str = "ja") -> str:
        """Transcribe a WAV file. Blocks — run via asyncio.to_thread()."""
        if self.loaded_model is None:
            raise RuntimeError(
                f"STT model '{self.active_key}' is not loaded. "
                "Load it first via POST /stt/load."
            )
        meta = MODELS[self.active_key]
        return meta["engine"].transcribe(self.loaded_model, wav_path, language)

    def load(self, key: str) -> dict:
        """Download + load model weights. Blocks — run via asyncio.to_thread().
        Returns {"elapsed_ms": int} on success, {"cancelled": True} if cancelled."""
        if key not in MODELS:
            raise ValueError(f"Unknown STT model key: {key}")
        meta = MODELS[key]
        backend = meta["backend"]

        if backend == "remote":
            self.loaded_model = None
            self.active_key = key
            return {"elapsed_ms": 0}

        if not AVAILABLE.get(backend, False):
            raise RuntimeError(
                f"Backend '{backend}' is not installed. "
                "Run: pip install " + ("mlx-audio" if backend == "mlx_audio" else "mlx-whisper")
            )

        self._cancel_flag.clear()
        t0 = time.monotonic()

        with self._load_lock:
            if self._cancel_flag.is_set():
                return {"cancelled": True}

            model = meta["engine"].load(meta["repo"])

            if self._cancel_flag.is_set():
                return {"cancelled": True}

            self.loaded_model = model
            self.active_key = key

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return {"elapsed_ms": elapsed_ms}

    def unload(self):
        """Release loaded model weights."""
        self.loaded_model = None

    def cancel_load(self):
        """Signal an in-progress load() to discard its result."""
        self._cancel_flag.set()


stt_manager = STTManager()
