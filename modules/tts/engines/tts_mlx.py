"""MLX-based Qwen3-TTS engine via mlx_audio."""
import asyncio
import logging
import os

import numpy as np

log = logging.getLogger("tts_mlx")

MODEL_ID    = os.getenv("MLX_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
SAMPLE_RATE = 24000   # Qwen3-TTS native output sample rate

_model = None


def _load():
    global _model
    if _model is not None:
        return
    try:
        from mlx_audio.tts import load
        _model = load(MODEL_ID)
        log.info(f"[tts_mlx] loaded {MODEL_ID}")
    except Exception as e:
        log.error(f"[tts_mlx] failed to load model: {e}")
        raise


def _is_available() -> bool:
    try:
        import mlx_audio  # noqa
        return True
    except ImportError:
        return False


AVAILABLE = _is_available()

if AVAILABLE:
    try:
        _load()
    except Exception:
        AVAILABLE = False


async def mlx_tts_stream(text: str, voice: str, language: str):
    """
    Async generator — yields raw float32 PCM chunks.
    Runs blocking inference in a thread pool.
    `language` is accepted for API compatibility but Qwen3-TTS infers it from text.
    """
    if not AVAILABLE or _model is None:
        raise RuntimeError("MLX TTS not available")

    loop = asyncio.get_event_loop()

    def _generate():
        return list(_model.generate(
            text=text,
            voice=voice,
            stream=True,
            streaming_interval=1.0,
        ))

    results = await loop.run_in_executor(None, _generate)

    for result in results:
        arr = np.asarray(result.audio, dtype=np.float32)
        yield arr.tobytes()
        await asyncio.sleep(0)
