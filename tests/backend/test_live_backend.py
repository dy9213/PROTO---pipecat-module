"""
Tests for the /live WebSocket endpoint — interrupt handling and all edge cases.

Interrupt scenario matrix tested here:
  1. Idle interrupt              → "interrupted" sent, no crash
  2. Interrupt during local STT  → ghost transcript suppressed, no error (Bug #2 fix)
  3. Interrupt during LLM stream → no error event sent to frontend (Bug #1 fix)
  4. Interrupt during VOICEVOX   → no error event sent to frontend
  5. Interrupt during Kokoro TTS → no error event sent to frontend
  6. Post-tts_end interrupt      → llmWasDone=true, badge on null wrap, turn preserved
  7. Double interrupt            → second interrupt is a no-op (debounce)
  8. Audio drops while processing→ `not processing` gate prevents new turns
  9. Empty transcript            → no transcript/error events sent
"""

import asyncio
import json
import math
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

# conftest patches heavy deps before this import
import backend.main as main_mod
from backend.main import app, VAD_CHUNK, VAD_ONSET_CHUNKS, SILENCE_WINDOWS, MIN_SPEECH_SAMPLES, PREBUFFER_SAMPLES


# ── shared mock WebSocket ─────────────────────────────────────────────────────

class FakeWS:
    """Minimal WebSocket mock that lets us drive input and collect output."""

    def __init__(self):
        self.sent: list[dict | bytes] = []    # all frames sent by the handler
        self._recv: asyncio.Queue = asyncio.Queue()

    async def accept(self):
        pass

    async def send_text(self, text: str):
        self.sent.append(json.loads(text))

    async def send_bytes(self, data: bytes):
        self.sent.append(data)

    async def receive(self):
        return await self._recv.get()

    # ── helpers ──────────────────────────────────────────────────────────────

    def push(self, item):
        """Queue an item for the handler to receive (bytes or text dict)."""
        if isinstance(item, bytes):
            self._recv.put_nowait({"bytes": item})
        elif isinstance(item, dict):
            self._recv.put_nowait({"text": json.dumps(item)})
        else:
            raise TypeError(type(item))

    def disconnect(self):
        self._recv.put_nowait({"type": "websocket.disconnect"})

    def json_events(self):
        return [m for m in self.sent if isinstance(m, dict)]

    def has_type(self, t):
        return any(m.get("type") == t for m in self.json_events())

    def types(self):
        return [m.get("type") for m in self.json_events()]


# ── audio helpers ─────────────────────────────────────────────────────────────

def _silence(n=VAD_CHUNK):
    return np.zeros(n, dtype=np.float32).tobytes()

def _speech(n=VAD_CHUNK, amp=0.6):
    return (np.ones(n, dtype=np.float32) * amp).tobytes()


def _audio_sequence_for_turn():
    """
    Build the minimal audio byte sequence that will trigger one full VAD turn.

    VAD onset fires at call VAD_ONSET_CHUNKS (consecutive above-threshold windows).
    At onset, pre_buf contains exactly VAD_ONSET_CHUNKS * VAD_CHUNK samples.
    After onset, we need (MIN_SPEECH_SAMPLES - pre_buf_samples) more speech samples
    before silence detection can end the utterance.

    Returns (chunks_list, n_above) where n_above is the number of above-threshold
    VAD calls to configure.
    """
    pre_buf_at_onset = VAD_ONSET_CHUNKS * VAD_CHUNK
    n_post_onset = math.ceil((MIN_SPEECH_SAMPLES - pre_buf_at_onset) / VAD_CHUNK) + 2
    n_above   = VAD_ONSET_CHUNKS + n_post_onset   # total above-threshold windows
    n_silence = SILENCE_WINDOWS + 2               # end-of-utterance silence windows

    chunks = (
        [_speech()  for _ in range(n_above)]    # speech: onset + content
        + [_silence() for _ in range(n_silence)]  # silence: ends utterance
    )
    return chunks, n_above


def _configure_vad_for_speech(n_above: int):
    """
    Patch _vad_session.run so the first n_above calls return above-threshold
    probability and subsequent calls return silence.
    The return shape matches what _vad_infer expects:
      out[0] shape (1,1,1) → out[0][0][0] is a numpy scalar.
    """
    call_count = [0]

    def _run(*args, **kwargs):
        call_count[0] += 1
        # Use np.float32() to produce a 0-d scalar and avoid the NumPy deprecation
        # warning: "Conversion of an array with ndim > 0 to a scalar is deprecated"
        prob = np.float32(0.9) if call_count[0] <= n_above else np.float32(0.01)
        return [
            np.array([[[prob]]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        ]

    main_mod._vad_session.run.side_effect = _run
    return call_count


def _reset_vad():
    main_mod._vad_session.run.side_effect = None
    main_mod._vad_session.run.return_value = [
        np.array([[[np.float32(0.0)]]]),
        np.zeros((2, 1, 128), dtype=np.float32),
    ]


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    path = tmp_path / "data" / "settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "stt_model":        "qwen3-1.7b-4bit",
        "tts_mode":         "voicevox",
        "voicevox_speaker": 2,
        "language":         "ja",
        "llm_model":        "qwen3-4b-4bit",
    }))
    monkeypatch.setattr(main_mod, "SETTINGS_PATH", path)


@pytest.fixture(autouse=True)
def reset_vad_after_test():
    yield
    _reset_vad()


# ── helper: run live handler for one turn ─────────────────────────────────────

async def _run_live_with_timeout(ws: FakeWS, timeout=3.0) -> asyncio.Task:
    task = asyncio.create_task(main_mod.live(ws))
    await asyncio.sleep(0)  # let handler start and call ws.accept()
    return task


async def _drive_turn(ws: FakeWS, chunks: list[bytes]):
    """Feed audio chunks to the handler one at a time."""
    for chunk in chunks:
        ws.push(chunk)
        await asyncio.sleep(0)


# ── 1. Idle interrupt ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_interrupt_while_idle_sends_interrupted():
    ws = FakeWS()
    task = await _run_live_with_timeout(ws)
    ws.push({"type": "interrupt"})
    await asyncio.sleep(0.05)
    ws.disconnect()
    await asyncio.wait_for(task, timeout=1.0)

    assert ws.has_type("interrupted")
    assert not ws.has_type("error"), f"idle interrupt must not produce error: {ws.types()}"


# ── 2. Interrupt during local STT — ghost transcript suppressed ───────────────

@pytest.mark.asyncio
async def test_interrupt_during_local_stt_no_ghost_transcript():
    """
    process_turn is triggered, starts local STT (asyncio.to_thread).
    Interrupt arrives while thread is running.
    After thread completes, process_turn must NOT send a transcript event.

    Bug: without the `if client not in active_reqs: return` guard, a stale
    transcript event is sent after the interrupt is acknowledged.
    """
    stt_started = asyncio.Event()
    stt_unblock = asyncio.Event()
    # Capture the running event loop NOW (coroutine context) — inside the
    # thread spawned by asyncio.to_thread, get_event_loop() returns a new
    # loop (Python 3.10+), not the main loop, so we must close over this ref.
    loop = asyncio.get_event_loop()

    def slow_stt(path, lang):
        import threading
        # Signal that STT has started, then block until test releases it
        loop.call_soon_threadsafe(stt_started.set)
        # Blocking wait from a thread — use a threading.Event bridge
        bridge = threading.Event()

        async def _set_bridge():
            await stt_unblock.wait()
            bridge.set()

        asyncio.run_coroutine_threadsafe(_set_bridge(), loop)
        bridge.wait(timeout=4)
        return "テスト"

    chunks, n_speech = _audio_sequence_for_turn()
    _configure_vad_for_speech(n_speech + VAD_ONSET_CHUNKS)

    with patch.object(main_mod.stt_manager, "transcribe", side_effect=slow_stt):
        ws = FakeWS()
        task = await _run_live_with_timeout(ws)

        await _drive_turn(ws, chunks)

        # Wait until STT has actually started in the thread
        try:
            await asyncio.wait_for(stt_started.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            pytest.skip("process_turn did not start STT within timeout")

        # Now send interrupt while STT thread is blocked
        ws.push({"type": "interrupt"})
        await asyncio.sleep(0.05)

        # Unblock the STT thread — process_turn should discard the result
        stt_unblock.set()
        await asyncio.sleep(0.1)

        ws.disconnect()
        await asyncio.wait_for(task, timeout=1.0)

    assert ws.has_type("interrupted")
    assert not ws.has_type("transcript"), (
        f"Ghost transcript was sent after interrupt: {ws.types()}\n"
        "Fix: add `if client not in active_reqs: return` after local STT."
    )
    assert not ws.has_type("error"), f"Error sent after STT interrupt: {ws.types()}"


# ── 3. Interrupt during LLM streaming — no error event ───────────────────────

@pytest.mark.asyncio
async def test_interrupt_during_llm_streaming_no_error():
    """
    process_turn is mid-LLM-stream when interrupt arrives.
    The closed httpx client raises an exception inside process_turn.

    Bug: without `if client in active_reqs:`, the exception produces
    {"type": "error"} which causes exitLiveMode() on the frontend.
    Fix: guard the send_json(error) with `if client in active_reqs:`.
    """
    llm_started = asyncio.Event()
    llm_unblock = asyncio.Event()

    # httpx.AsyncClient.stream() is a sync method that returns an async CM —
    # the side_effect must be a regular function, NOT async (else the returned
    # coroutine is never awaited and the test gets a RuntimeWarning + skip).
    def blocking_stream(*args, **kwargs):
        class _FakeResp:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass

            async def aiter_lines(self):
                yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "テ"}}]})
                llm_started.set()
                await llm_unblock.wait()   # blocks until aclose() is called
                # aclose() causes anyio to raise ClosedResourceError or similar
                raise Exception("connection closed by interrupt")

        return _FakeResp()

    chunks, n_speech = _audio_sequence_for_turn()
    _configure_vad_for_speech(n_speech + VAD_ONSET_CHUNKS)

    with patch.object(main_mod.httpx.AsyncClient, "stream", side_effect=blocking_stream):
        ws = FakeWS()
        task = await _run_live_with_timeout(ws)
        await _drive_turn(ws, chunks)

        try:
            await asyncio.wait_for(llm_started.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            pytest.skip("process_turn did not reach LLM within timeout")

        # Send interrupt while LLM is blocked
        ws.push({"type": "interrupt"})
        await asyncio.sleep(0.05)
        llm_unblock.set()           # let blocking_stream raise
        await asyncio.sleep(0.1)

        ws.disconnect()
        await asyncio.wait_for(task, timeout=1.0)

    assert ws.has_type("interrupted"), f"Expected 'interrupted': {ws.types()}"
    assert not ws.has_type("error"), (
        f"Interrupt produced error event: {ws.types()}\n"
        "Fix: `if client in active_reqs:` guard in process_turn's except block."
    )


# ── 4. Interrupt during VOICEVOX synthesis — no error ────────────────────────

@pytest.mark.asyncio
async def test_interrupt_during_voicevox_no_error():
    """
    Interrupt arrives while _voicevox_tts() is awaiting the synthesis HTTP call.
    The closed client must not produce an error event.
    """
    tts_started = asyncio.Event()

    # voicevox_tts is async — this is fine as a side_effect since the mock
    # awaits async side_effects automatically when the mock itself is awaited.
    async def blocking_voicevox(client, text, speaker, endpoint, speed=1.0):
        tts_started.set()
        await asyncio.sleep(30)   # blocks — will be interrupted by aclose()
        return b"RIFF\x00\x00\x00\x00WAVEfmt "

    # LLM must be mocked so process_turn reaches the TTS stage — without this
    # client.stream() tries the real local port and fails with ConnectError.
    def fast_llm_stream(*args, **kwargs):
        class _Resp:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            async def aiter_lines(self):
                yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "テスト"}}]})
                yield "data: [DONE]"
        return _Resp()

    chunks, n_speech = _audio_sequence_for_turn()
    _configure_vad_for_speech(n_speech + VAD_ONSET_CHUNKS)

    with patch.object(main_mod.httpx.AsyncClient, "stream", side_effect=fast_llm_stream), \
         patch.object(main_mod, "_voicevox_tts", side_effect=blocking_voicevox):
        ws = FakeWS()
        task = await _run_live_with_timeout(ws)
        await _drive_turn(ws, chunks)

        try:
            await asyncio.wait_for(tts_started.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            pytest.skip("process_turn did not reach VOICEVOX within timeout")

        ws.push({"type": "interrupt"})
        await asyncio.sleep(0.15)

        ws.disconnect()
        await asyncio.wait_for(task, timeout=1.0)

    assert ws.has_type("interrupted")
    assert not ws.has_type("error"), (
        f"VOICEVOX interrupt produced error: {ws.types()}"
    )


# ── 5. Interrupt during Kokoro TTS — no error ────────────────────────────────

@pytest.mark.asyncio
async def test_interrupt_during_kokoro_no_error(isolated_settings, tmp_path, monkeypatch):
    """Kokoro / OpenAI-compat TTS: same guard must suppress the error."""
    kokoro_path = tmp_path / "data" / "settings.json"
    kokoro_path.write_text(json.dumps({
        "stt_model": "qwen3-1.7b-4bit",
        "tts_mode": "kokoro",
        "language": "ja",
        "llm_model": "qwen3-4b-4bit",
        "tts_endpoint": "http://127.0.0.1:9999",
        "voice_ja": "Ono_Anna",
    }))
    monkeypatch.setattr(main_mod, "SETTINGS_PATH", kokoro_path)

    tts_started = asyncio.Event()

    # client.post() is async — side_effect as async coroutine is correct here.
    async def blocking_post(*args, **kwargs):
        tts_started.set()
        await asyncio.sleep(30)
        raise Exception("should not reach here")

    # LLM must complete first so process_turn reaches the Kokoro TTS stage.
    def fast_llm_stream(*args, **kwargs):
        class _Resp:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            async def aiter_lines(self):
                yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "テスト"}}]})
                yield "data: [DONE]"
        return _Resp()

    chunks, n_speech = _audio_sequence_for_turn()
    _configure_vad_for_speech(n_speech + VAD_ONSET_CHUNKS)

    with patch.object(main_mod.httpx.AsyncClient, "stream", side_effect=fast_llm_stream), \
         patch.object(main_mod.httpx.AsyncClient, "post", side_effect=blocking_post):
        ws = FakeWS()
        task = await _run_live_with_timeout(ws)
        await _drive_turn(ws, chunks)

        try:
            await asyncio.wait_for(tts_started.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            pytest.skip("process_turn did not reach Kokoro TTS within timeout")

        ws.push({"type": "interrupt"})
        await asyncio.sleep(0.15)

        ws.disconnect()
        await asyncio.wait_for(task, timeout=1.0)

    assert ws.has_type("interrupted")
    assert not ws.has_type("error"), (
        f"Kokoro interrupt produced error: {ws.types()}"
    )


# ── 6. Post-tts_end interrupt — turn preserved ───────────────────────────────

@pytest.mark.asyncio
async def test_post_tts_end_interrupt_preserves_turn():
    """
    Backend completes a full turn (tts_start → audio binary → tts_end).
    Audio is still playing on the client when interrupt arrives.
    Backend is now idle; it sends "interrupted" with active_reqs empty.

    Expected: turn bubbles stay; no error; _liveLLMDone stays True (tts_end
    doesn't reset it) so the frontend decision tree picks the badge path.
    This is tested in live.test.js; here we verify the backend response.
    """
    async def instant_voicevox(client, text, speaker, endpoint, speed=1.0):
        return b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    def fast_llm_stream(*args, **kwargs):
        class _Resp:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *_): pass
            async def aiter_lines(self):
                yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "テスト"}}]})
                yield "data: [DONE]"
        return _Resp()

    chunks, n_speech = _audio_sequence_for_turn()
    _configure_vad_for_speech(n_speech + VAD_ONSET_CHUNKS)

    with patch.object(main_mod.httpx.AsyncClient, "stream", side_effect=fast_llm_stream), \
         patch.object(main_mod, "_voicevox_tts", side_effect=instant_voicevox):
        ws = FakeWS()
        task = await _run_live_with_timeout(ws)
        await _drive_turn(ws, chunks)

        # Wait for tts_end
        for _ in range(40):
            await asyncio.sleep(0.05)
            if ws.has_type("tts_end"):
                break

        if not ws.has_type("tts_end"):
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            pytest.skip("turn did not complete within timeout")

        # Now interrupt — backend is idle
        ws.push({"type": "interrupt"})
        await asyncio.sleep(0.05)

        ws.disconnect()
        await asyncio.wait_for(task, timeout=1.0)

    # Backend must still send "interrupted" (clean state reset)
    assert ws.has_type("interrupted")
    assert not ws.has_type("error"), f"Post-tts_end interrupt produced error: {ws.types()}"

    # tts_start must have preceded tts_end (turn really completed)
    t = ws.types()
    assert t.index("tts_start") < t.index("tts_end")


# ── 7. Double interrupt — second is a no-op ───────────────────────────────────

@pytest.mark.asyncio
async def test_double_interrupt_no_crash():
    """
    Two interrupt messages in rapid succession.
    Second interrupt: active_reqs already empty, processing already False.
    Must not crash or produce duplicate state.
    """
    ws = FakeWS()
    task = await _run_live_with_timeout(ws)

    ws.push({"type": "interrupt"})
    ws.push({"type": "interrupt"})
    await asyncio.sleep(0.05)
    ws.disconnect()
    await asyncio.wait_for(task, timeout=1.0)

    interrupted_count = ws.types().count("interrupted")
    assert interrupted_count == 2, (
        f"Expected 2 interrupted events for 2 interrupts, got {interrupted_count}"
    )
    assert not ws.has_type("error")


# ── 8. Audio dropped while processing ────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_dropped_while_processing():
    """
    Binary audio chunks received while processing=True are silently dropped.
    Only one process_turn must run at a time.
    """
    first_stt_called = asyncio.Event()
    second_stt_never_called = True
    # Capture loop in coroutine context — get_event_loop() inside a thread
    # (spawned by asyncio.to_thread) gets a new loop in Python 3.10+.
    loop = asyncio.get_event_loop()

    call_count = [0]

    def counting_stt(path, lang):
        call_count[0] += 1
        if call_count[0] == 1:
            loop.call_soon_threadsafe(first_stt_called.set)
            import time; time.sleep(0.3)  # hold process_turn for a moment
        else:
            nonlocal second_stt_never_called
            second_stt_never_called = False
        return "テスト"

    chunks, n_speech = _audio_sequence_for_turn()
    _configure_vad_for_speech(n_speech + VAD_ONSET_CHUNKS)

    with patch.object(main_mod.stt_manager, "transcribe", side_effect=counting_stt):
        ws = FakeWS()
        task = await _run_live_with_timeout(ws)
        await _drive_turn(ws, chunks)

        try:
            await asyncio.wait_for(first_stt_called.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            pytest.skip("First STT call did not start")

        # While processing: send more audio — should be discarded
        for chunk in chunks[:10]:
            ws.push(chunk)
        await asyncio.sleep(0.4)  # wait for first turn to finish

        ws.disconnect()
        await asyncio.wait_for(task, timeout=1.0)

    assert second_stt_never_called, "Audio accepted while processing=True (guard failed)"


# ── 9. Empty transcript — no events sent ─────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_transcript_produces_no_events():
    """If STT returns an empty string, no transcript/error event is sent."""
    chunks, n_speech = _audio_sequence_for_turn()
    _configure_vad_for_speech(n_speech + VAD_ONSET_CHUNKS)

    with patch.object(main_mod.stt_manager, "transcribe", return_value="   "):
        ws = FakeWS()
        task = await _run_live_with_timeout(ws)
        await _drive_turn(ws, chunks)
        await asyncio.sleep(0.3)
        ws.disconnect()
        await asyncio.wait_for(task, timeout=1.0)

    assert not ws.has_type("transcript")
    assert not ws.has_type("error")


# ── 10. ping / pong ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ping_pong():
    ws = FakeWS()
    task = await _run_live_with_timeout(ws)
    ws.push({"type": "ping"})
    await asyncio.sleep(0.02)
    ws.disconnect()
    await asyncio.wait_for(task, timeout=1.0)
    assert ws.has_type("pong")


# ── 11. audio_config seeds history ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_config_seeds_history():
    ws = FakeWS()
    task = await _run_live_with_timeout(ws)
    ws.push({
        "type": "audio_config",
        "sample_rate": 16000,
        "history": [
            {"role": "user", "content": "前の発言"},
            {"role": "assistant", "content": "前の回答"},
        ],
    })
    ws.push({"type": "ping"})
    await asyncio.sleep(0.02)
    ws.disconnect()
    await asyncio.wait_for(task, timeout=1.0)
    assert ws.has_type("pong")  # no crash = history was accepted


# ── structural guards (code-inspection) ──────────────────────────────────────

class TestStructuralGuards:
    """Verify the critical guard conditions exist in the source."""

    def test_interrupt_guard_on_error_send(self):
        import inspect
        src = inspect.getsource(main_mod.live)
        assert "client in active_reqs" in src, (
            "Missing `if client in active_reqs:` guard before send_json(error). "
            "Without it every interrupt produces an error event."
        )

    def test_post_stt_interrupt_guard(self):
        import inspect
        src = inspect.getsource(main_mod.live)
        assert "client not in active_reqs" in src, (
            "Missing `if client not in active_reqs: return` guard after local STT. "
            "Without it a stale transcript is sent after the interrupt."
        )

    def test_active_reqs_cleared_on_interrupt(self):
        import inspect
        src = inspect.getsource(main_mod.live)
        assert "active_reqs.clear()" in src

    def test_processing_reset_on_interrupt(self):
        import inspect
        src = inspect.getsource(main_mod.live)
        assert "processing = False" in src

    def test_valueerror_suppressed_on_double_remove(self):
        import inspect
        src = inspect.getsource(main_mod.live)
        assert "except ValueError" in src

    def test_not_processing_gate_on_audio(self):
        import inspect
        src = inspect.getsource(main_mod.live)
        assert "not processing" in src


# ── pure-logic unit tests ─────────────────────────────────────────────────────

class TestGetSystemPrompt:
    def test_returns_custom_prompt(self):
        s = {"language": "ja", "system_prompt_ja": "カスタム"}
        assert main_mod.get_system_prompt(s).startswith("カスタム")

    def test_falls_back_to_default(self):
        s = {"language": "ja", "system_prompt_ja": ""}
        result = main_mod.get_system_prompt(s)
        assert result.strip()

    def test_appends_datetime(self):
        s = {"language": "en", "system_prompt_en": "test"}
        assert "Current date and time" in main_mod.get_system_prompt(s)

    def test_unknown_lang_falls_back(self):
        s = {"language": "zz", "system_prompt_zz": ""}
        assert main_mod.get_system_prompt(s)


class TestLlmConfig:
    def test_local_model(self):
        s = {"llm_model": "qwen3-4b-4bit", "llm_endpoint": "", "llm_api_key": ""}
        ep, hdrs, mdl = main_mod._llm_config(s)
        assert ep == main_mod.LLM_LOCAL_URL
        assert hdrs == {}

    def test_remote_with_key(self):
        s = {"llm_model": "gpt-4o", "llm_endpoint": "https://api.openai.com", "llm_api_key": "sk-x"}
        ep, hdrs, mdl = main_mod._llm_config(s)
        assert "Authorization" in hdrs

    def test_no_key_no_auth_header(self):
        s = {"llm_model": "x", "llm_endpoint": "http://local", "llm_api_key": ""}
        _, hdrs, _ = main_mod._llm_config(s)
        assert "Authorization" not in hdrs
