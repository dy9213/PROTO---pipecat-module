"""Remote STT sentinel — transcription is handled by main.py's remote path."""

AVAILABLE: bool = True  # always "available"; requires endpoint to be configured


def load(repo: str) -> None:
    return None


def transcribe(model: None, wav_path: str, language: str = "ja") -> str:
    raise RuntimeError("Remote STT transcription is handled directly by the /stt endpoint.")
