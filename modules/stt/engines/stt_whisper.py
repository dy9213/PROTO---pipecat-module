"""Kotoba-Whisper STT engine via mlx_whisper."""

AVAILABLE: bool = False
try:
    import mlx_whisper  # noqa: F401
    AVAILABLE = True
except ImportError:
    pass


def load(repo: str) -> dict:
    # mlx_whisper downloads weights on first transcribe call.
    # Store a sentinel so the manager knows it's "loaded".
    return {"__mlx_whisper_ready__": True, "repo": repo}


def transcribe(model: dict, wav_path: str, language: str = "ja") -> str:
    import mlx_whisper as _mlx_whisper
    result = _mlx_whisper.transcribe(wav_path, path_or_hf_repo=model["repo"], language=language)
    return result["text"]
