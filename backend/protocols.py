"""
Abstract engine interfaces for STT, TTS, and LLM.
Each capability manager and engine module implicitly satisfies one of these protocols.
Swap an engine by replacing which concrete module is wired in the manager.
"""
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class STTEngine(Protocol):
    AVAILABLE: bool

    def load(self, repo: str) -> Any: ...
    def transcribe(self, model: Any, wav_path: str, language: str) -> str: ...


@runtime_checkable
class TTSEngine(Protocol):
    AVAILABLE: bool

    async def synthesize(self, text: str, **kwargs) -> bytes: ...


@runtime_checkable
class LLMEngine(Protocol):
    AVAILABLE: bool

    async def chat_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[str]: ...
