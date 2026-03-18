"""VOICEVOX remote TTS engine via HTTP."""
import httpx

AVAILABLE: bool = True  # always available; requires endpoint to be configured

_voicevox_cancellable: bool | None = None  # None=untested, True/False=cached


async def voicevox_tts(client: httpx.AsyncClient, text: str, speaker: int, endpoint: str) -> bytes:
    """Synthesize speech via VOICEVOX HTTP API.
    Prefers /cancellable_synthesis (requires --enable_cancellable_synthesis --init_processes 2).
    Falls back to /synthesis on 404 and caches the result."""
    global _voicevox_cancellable
    base = endpoint.rstrip("/")
    r1 = await client.post(f"{base}/audio_query", params={"text": text, "speaker": speaker})
    r1.raise_for_status()

    if _voicevox_cancellable is not False:
        r2 = await client.post(f"{base}/cancellable_synthesis", params={"speaker": speaker}, json=r1.json())
        if r2.status_code == 404:
            _voicevox_cancellable = False
            print("[voicevox] /cancellable_synthesis not available — falling back to /synthesis. "
                  "Launch VOICEVOX with --enable_cancellable_synthesis for faster interrupts.", flush=True)
        else:
            r2.raise_for_status()
            _voicevox_cancellable = True
            return r2.content

    r2 = await client.post(f"{base}/synthesis", params={"speaker": speaker}, json=r1.json())
    r2.raise_for_status()
    return r2.content
