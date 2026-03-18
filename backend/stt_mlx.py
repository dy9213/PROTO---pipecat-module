"""
MLX-native STT backend for local Japanese ASR.

Supports Qwen3-ASR variants (via mlx_audio) and Kotoba-Whisper (via mlx_whisper).
The "remote" key is a sentinel — transcription is handled by main.py's remote path.
"""
import threading
import time
from typing import Any

MODELS = {
    "qwen3-0.6b-4bit": {"repo": "mlx-community/Qwen3-ASR-0.6B-4bit",  "backend": "mlx_audio",   "label": "Qwen3 0.6B 4bit"},
    "qwen3-0.6b-8bit": {"repo": "mlx-community/Qwen3-ASR-0.6B-8bit",  "backend": "mlx_audio",   "label": "Qwen3 0.6B 8bit"},
    "qwen3-0.6b-bf16": {"repo": "mlx-community/Qwen3-ASR-0.6B-bf16",  "backend": "mlx_audio",   "label": "Qwen3 0.6B bf16"},
    "qwen3-1.7b-4bit": {"repo": "mlx-community/Qwen3-ASR-1.7B-4bit",  "backend": "mlx_audio",   "label": "Qwen3 1.7B 4bit"},
    "qwen3-1.7b-8bit": {"repo": "mlx-community/Qwen3-ASR-1.7B-8bit",  "backend": "mlx_audio",   "label": "Qwen3 1.7B 8bit"},
    "qwen3-1.7b-bf16": {"repo": "mlx-community/Qwen3-ASR-1.7B-bf16",  "backend": "mlx_audio",   "label": "Qwen3 1.7B bf16"},
    "kotoba-whisper":  {"repo": "kotoba-tech/kotoba-whisper-v2.0",      "backend": "mlx_whisper", "label": "Kotoba Whisper v2"},
    "remote":          {"repo": None,                                    "backend": "remote",      "label": "Remote (OpenAI-compat)"},
}
DEFAULT_MODEL = "qwen3-0.6b-4bit"

# ISO 639-1 → full language name expected by Qwen3-ASR
_LANG_CODE_TO_NAME: dict[str, str] = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "tr": "Turkish",
}

# Checked at import time — backends are optional
AVAILABLE: dict[str, bool] = {"mlx_audio": False, "mlx_whisper": False}
try:
    import mlx_audio  # noqa: F401
    AVAILABLE["mlx_audio"] = True
except ImportError:
    pass
try:
    import mlx_whisper  # noqa: F401
    AVAILABLE["mlx_whisper"] = True
except ImportError:
    pass


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
        backend = meta["backend"]
        repo = meta["repo"]

        if backend == "mlx_audio":
            from mlx_audio.stt.generate import generate_transcription
            lang_name = _LANG_CODE_TO_NAME.get(language.lower(), language)
            result = generate_transcription(self.loaded_model, audio=wav_path, language=lang_name)
            return result.text

        if backend == "mlx_whisper":
            import mlx_whisper as _mlx_whisper
            result = _mlx_whisper.transcribe(wav_path, path_or_hf_repo=repo, language=language)
            return result["text"]

        raise RuntimeError(f"Cannot transcribe with backend '{backend}'")

    def load(self, key: str) -> dict:
        """Download + load model weights. Blocks — run via asyncio.to_thread().
        Returns {"elapsed_ms": int} on success, {"cancelled": True} if cancelled."""
        if key not in MODELS:
            raise ValueError(f"Unknown STT model key: {key}")
        meta = MODELS[key]
        backend = meta["backend"]
        repo = meta["repo"]

        if backend == "remote":
            # No load needed for remote backend — just update active key
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

            if backend == "mlx_audio":
                from mlx_audio.stt.utils import load_model
                model = load_model(repo)
            elif backend == "mlx_whisper":
                # mlx_whisper downloads weights on first transcribe call.
                # Store a sentinel so transcribe() knows it's "loaded".
                model = {"__mlx_whisper_ready__": True}
            else:
                raise RuntimeError(f"Unknown backend: {backend}")

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
