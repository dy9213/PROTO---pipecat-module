"""
TTS capability manager.
Engines:
  voicevox  — local subprocess (voicevox_engine binary, port 50021)
  kokoro    — remote OpenAI-compat HTTP endpoint
"""
from modules.tts.engines.tts_remote import voicevox_tts

__all__ = ["voicevox_tts"]
