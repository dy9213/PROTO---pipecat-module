"""
TTS capability manager — thin router for now.
Full engine abstraction added when voicevox_core (local) lands.
"""
from modules.tts.engines.tts_mlx    import mlx_tts_stream, AVAILABLE as MLX_AVAILABLE, SAMPLE_RATE as MLX_SAMPLE_RATE
from modules.tts.engines.tts_remote import voicevox_tts

__all__ = ["mlx_tts_stream", "MLX_AVAILABLE", "MLX_SAMPLE_RATE", "voicevox_tts"]
