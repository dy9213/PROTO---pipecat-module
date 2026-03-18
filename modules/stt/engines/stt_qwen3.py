"""Qwen3-ASR STT engine via mlx_audio."""
from typing import Any

AVAILABLE: bool = False
try:
    import mlx_audio  # noqa: F401
    AVAILABLE = True
except ImportError:
    pass

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


def load(repo: str) -> Any:
    from mlx_audio.stt.utils import load_model
    return load_model(repo)


def transcribe(model: Any, wav_path: str, language: str = "ja") -> str:
    from mlx_audio.stt.generate import generate_transcription
    lang_name = _LANG_CODE_TO_NAME.get(language.lower(), language)
    result = generate_transcription(model, audio=wav_path, language=lang_name)
    return result.text
