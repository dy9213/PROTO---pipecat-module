import asyncio, json, os, subprocess, tempfile, urllib.request
from pathlib import Path
from typing import AsyncGenerator

import httpx
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# ── debug ──────────────────────────────────────────────────────────────────────
DEBUG_RAW_SSE = False   # set True to print every raw SSE line from the LLM
DEBUG_VAD     = False   # set True to print VAD prob + silence window counts

# ── constants ──────────────────────────────────────────────────────────────────
VAD_THRESHOLD      = 0.4    # Silero speech probability threshold
VAD_ONSET_CHUNKS   = 2      # consecutive above-threshold windows to confirm onset (~64ms)
VAD_SILENCE_MS     = 1200   # ms of silence after speech to end utterance
VAD_MIN_SPEECH_MS  = 300    # discard utterances shorter than this
VAD_PREBUFFER_MS   = 200    # pre-roll before onset
SAMPLE_RATE        = 16000
VAD_CHUNK          = 512    # samples per Silero window (32ms at 16kHz)
SILENCE_WINDOWS    = int(SAMPLE_RATE * VAD_SILENCE_MS / 1000 / VAD_CHUNK)
MIN_SPEECH_SAMPLES = int(SAMPLE_RATE * VAD_MIN_SPEECH_MS / 1000)
PREBUFFER_SAMPLES  = int(SAMPLE_RATE * VAD_PREBUFFER_MS / 1000)

# ── Silero VAD ─────────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).parent.parent / "data" / "silero_vad.onnx"
_MODEL_URL  = "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx"

if not _MODEL_PATH.exists():
    print("Downloading Silero VAD model…")
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("Silero VAD model downloaded.")

_vad_session = ort.InferenceSession(str(_MODEL_PATH), providers=["CPUExecutionProvider"])
print(f"[vad] inputs: {[i.name for i in _vad_session.get_inputs()]}")

def _vad_infer(samples: np.ndarray, state: np.ndarray):
    """Run one 512-sample Silero window. Returns (prob, new_state).
    samples: float32 array in [-1, 1], shape (512,)
    state:   float32 array shape (2, 1, 128)
    """
    out = _vad_session.run(None, {
        "input": samples[np.newaxis, :].astype(np.float32),
        "sr":    np.array([SAMPLE_RATE], dtype=np.int64),
        "state": state,
    })
    return float(out[0][0][0]), out[1]

SETTINGS_PATH = Path(__file__).parent.parent / "data" / "settings.json"
DEFAULT_SETTINGS = {
    "stt_endpoint":    "",
    "llm_endpoint":    "",
    "tts_endpoint":    "",
    "llm_model":       "",
    "llm_api_key":     "",
    "language":        "en",
    "voice_en":        "Ryan",
    "voice_ja":        "Ono_Anna",
    "voice_zh":        "Vivian",
    "system_prompt_en": "",
    "system_prompt_ja": "",
    "system_prompt_zh": "",
}
TTS_LANGUAGE_MAP = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
}

DEFAULT_SYSTEM_PROMPTS = {
    "en": "You are a helpful voice assistant. Respond concisely in English.",
    "ja": "あなたは役立つ音声アシスタントです。日本語で簡潔に回答してください。",
    "zh": "你是一个有用的语音助手。请用中文简洁地回答。",
}

def get_system_prompt(s: dict) -> str:
    lang = s.get("language", "en")
    key  = f"system_prompt_{lang}"
    sp   = s.get(key, "").strip()
    return sp if sp else DEFAULT_SYSTEM_PROMPTS.get(lang, DEFAULT_SYSTEM_PROMPTS["en"])

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def load_settings() -> dict:
    try:
        return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_PATH.read_text())}
    except Exception:
        return DEFAULT_SETTINGS.copy()

def save_settings(s: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2))

# ── helpers ────────────────────────────────────────────────────────────────────
def webm_to_wav(data: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src:
        src.write(data); src_path = src.name
    dst_path = src_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", dst_path],
            check=True, capture_output=True,
        )
        return Path(dst_path).read_bytes()
    finally:
        os.unlink(src_path)
        if os.path.exists(dst_path): os.unlink(dst_path)

def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, samples, sample_rate, subtype="PCM_16")
        return Path(f.name).read_bytes()

# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/settings")
async def get_settings():
    return load_settings()

class SettingsIn(BaseModel):
    stt_endpoint:  str = ""
    llm_endpoint:  str = ""
    tts_endpoint:  str = ""
    llm_model:     str = ""
    llm_api_key:   str = ""
    language:      str = "en"
    voice_en:        str = "af_heart"
    voice_ja:        str = "jf_alpha"
    voice_zh:        str = "zf_xiaobei"
    system_prompt_en: str = ""
    system_prompt_ja: str = ""
    system_prompt_zh: str = ""

@app.post("/settings")
async def post_settings(body: SettingsIn):
    s = body.model_dump()
    save_settings(s)
    status = {"stt": False, "llm": False, "tts": False}
    async with httpx.AsyncClient(timeout=3) as c:
        for key, url, path in [
            ("stt", s["stt_endpoint"], "/health"),
            ("llm", s["llm_endpoint"], "/v1/models"),
            ("tts", s["tts_endpoint"], "/health"),
        ]:
            if not url: continue
            try:
                r = await c.get(url.rstrip("/") + path)
                status[key] = r.status_code < 400
            except Exception:
                pass
    return status

@app.post("/stt")
async def stt(audio: UploadFile = File(...), language: str = Form("en")):
    s = load_settings()
    raw = await audio.read()
    is_wav = (audio.content_type == "audio/wav" or (audio.filename or "").endswith(".wav"))
    wav = raw if is_wav else webm_to_wav(raw)
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            s["stt_endpoint"].rstrip("/") + "/v1/audio/transcriptions",
            files={"file": ("audio.wav", wav, "audio/wav")},
            data={"model": "whisper-1", "language": language},
        )
        r.raise_for_status()
    return {"transcript": r.json().get("text", "")}

class ChatIn(BaseModel):
    message: str
    history: list = []
    language: str = "en"

@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    s = load_settings()
    messages = [{"role": "system", "content": get_system_prompt(s)}]
    messages += body.history[-10:]
    messages.append({"role": "user", "content": body.message})

    async def stream() -> AsyncGenerator[bytes, None]:
        # connect=10: fail fast if LM Studio is unreachable
        # read=30: max silence between tokens before giving up
        headers = {"Authorization": f"Bearer {s['llm_api_key']}"} if s.get("llm_api_key") else {}
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=30, write=10, pool=5)) as c:
            async with c.stream("POST", s["llm_endpoint"].rstrip("/") + "/api/chat/completions",
                                 headers=headers,
                                 json={"model": s.get("llm_model") or "local-model", "messages": messages, "stream": True,
                                       "tool_ids": ["local_web_search"]}) as r:
                async for line in r.aiter_lines():
                    if DEBUG_RAW_SSE:
                        print(f"[sse/chat] {line}", flush=True)
                    if line.startswith("data: "):
                        yield (line + "\n\n").encode()

    return StreamingResponse(stream(), media_type="text/event-stream")

class TTSIn(BaseModel):
    text: str
    language: str = "en"

@app.post("/tts")
async def tts(body: TTSIn):
    s = load_settings()
    voice = s.get(f"voice_{body.language}", s["voice_en"])
    tts_lang = TTS_LANGUAGE_MAP.get(body.language, "English")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            s["tts_endpoint"].rstrip("/") + "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": body.text, "voice": voice, "language": tts_lang, "response_format": "wav"},
        )
        r.raise_for_status()
    return Response(content=r.content, media_type="audio/wav")

@app.post("/shutdown")
async def shutdown():
    asyncio.get_event_loop().call_later(0.1, os._exit, 0)
    return {"status": "shutting down"}

# ── live WebSocket ─────────────────────────────────────────────────────────────
@app.websocket("/live")
async def live(ws: WebSocket):
    await ws.accept()
    s = load_settings()

    # VAD state
    vad_state = np.zeros((2, 1, 128), dtype=np.float32)
    vad_buf: list[float] = []   # accumulate float32 samples until VAD_CHUNK
    _vad_last_log = 0.0

    # utterance state
    pre_buf: list[bytes] = []
    pre_buf_samples = 0
    speech_buf = bytearray()
    speech_samples = 0
    silence_windows = 0
    onset_streak = 0
    in_speech = False

    processing = False
    active_reqs: list[httpx.AsyncClient] = []
    conv_messages: list[dict] = []   # seeded from audio_config history, grows each turn

    async def send_json(obj): await ws.send_text(json.dumps(obj))

    async def process_turn(pcm: bytes):
        nonlocal processing
        processing = True
        client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=30, write=10, pool=5))
        active_reqs.append(client)
        try:
            wav = pcm_to_wav(pcm)
            r = await client.post(
                s["stt_endpoint"].rstrip("/") + "/v1/audio/transcriptions",
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": "whisper-1", "language": s["language"]},
            )
            r.raise_for_status()
            transcript = r.json().get("text", "").strip()
            if not transcript:
                return
            await send_json({"type": "transcript", "text": transcript, "final": True})

            conv_messages.append({"role": "user", "content": transcript})
            messages = [{"role": "system", "content": get_system_prompt(s)}] + conv_messages[-20:]

            full_text = ""
            llm_headers = {"Authorization": f"Bearer {s['llm_api_key']}"} if s.get("llm_api_key") else {}
            async with client.stream("POST", s["llm_endpoint"].rstrip("/") + "/api/chat/completions",
                                     headers=llm_headers,
                                     json={"model": s.get("llm_model") or "local-model", "stream": True,
                                           "messages": messages, "tool_ids": ["local_web_search"]}) as resp:
                async for line in resp.aiter_lines():
                    if DEBUG_RAW_SSE:
                        print(f"[sse/live] {line}", flush=True)
                    if not line.startswith("data: "): continue
                    data = line[6:]
                    if data == "[DONE]": break
                    try:
                        chunk = json.loads(data)
                        tok = chunk["choices"][0]["delta"].get("content", "")
                        if tok:
                            full_text += tok
                            await send_json({"type": "token", "text": tok, "agent": "assistant"})
                    except Exception:
                        pass

            if full_text:
                conv_messages.append({"role": "assistant", "content": full_text})
                voice = s.get(f"voice_{s['language']}", s["voice_en"])
                tts_lang = TTS_LANGUAGE_MAP.get(s["language"], "English")
                await send_json({"type": "tts_start"})
                r2 = await client.post(
                    s["tts_endpoint"].rstrip("/") + "/v1/audio/speech",
                    json={"model": "qwen3-tts", "input": full_text, "voice": voice, "language": tts_lang, "response_format": "wav"},
                )
                r2.raise_for_status()
                await ws.send_bytes(r2.content)
                await send_json({"type": "tts_end"})
        except Exception as e:
            await send_json({"type": "error", "message": str(e)})
        finally:
            processing = False
            active_reqs.remove(client)
            await client.aclose()

    audio_chunks_received = 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "text" in msg:
                data = json.loads(msg["text"])
                if data.get("type") == "interrupt":
                    for c in active_reqs:
                        await c.aclose()
                    active_reqs.clear()
                    processing = False
                    in_speech = False
                    speech_buf = bytearray()
                    speech_samples = 0
                    silence_windows = 0
                    onset_streak = 0
                    pre_buf.clear()
                    pre_buf_samples = 0
                    vad_buf.clear()
                    vad_state = np.zeros((2, 1, 128), dtype=np.float32)
                    await send_json({"type": "interrupted"})
                elif data.get("type") == "ping":
                    await send_json({"type": "pong"})
                elif data.get("type") == "audio_config":
                    for m in data.get("history", []):
                        if m.get("role") in ("user", "assistant") and m.get("content"):
                            conv_messages.append({"role": m["role"], "content": m["content"]})
            elif "bytes" in msg and not processing:
                chunk = msg["bytes"]
                audio_chunks_received += 1
                if audio_chunks_received <= 3:
                    print(f"[live] binary chunk #{audio_chunks_received} len={len(chunk)}", flush=True)
                samples_f = np.frombuffer(chunk, dtype=np.float32)
                vad_buf.extend(samples_f.tolist())

                # always accumulate raw bytes for speech_buf / pre_buf
                n = len(samples_f)
                if not in_speech:
                    pre_buf.append(chunk)
                    pre_buf_samples += n
                    while pre_buf_samples - len(pre_buf[0]) // 4 > PREBUFFER_SAMPLES:
                        pre_buf_samples -= len(pre_buf[0]) // 4
                        pre_buf.pop(0)
                else:
                    speech_buf.extend(chunk)
                    speech_samples += n

                # Silero VAD on each full 512-sample window
                while len(vad_buf) >= VAD_CHUNK:
                    window = np.array(vad_buf[:VAD_CHUNK], dtype=np.float32)
                    vad_buf = vad_buf[VAD_CHUNK:]
                    prob, vad_state = _vad_infer(window, vad_state)
                    above = prob >= VAD_THRESHOLD
                    if DEBUG_VAD:
                        _now = asyncio.get_event_loop().time()
                        if _now - _vad_last_log >= 0.5:
                            _vad_last_log = _now
                            print(f"[vad] prob={prob:.3f} above={above} in_speech={in_speech} silence={silence_windows}/{SILENCE_WINDOWS} onset={onset_streak}/{VAD_ONSET_CHUNKS}", flush=True)

                    if not in_speech:
                        if above:
                            onset_streak += 1
                        else:
                            onset_streak = 0

                        if onset_streak >= VAD_ONSET_CHUNKS:
                            in_speech = True
                            onset_streak = 0
                            silence_windows = 0
                            speech_buf = bytearray()
                            for c in pre_buf:
                                speech_buf.extend(c)
                            speech_samples = pre_buf_samples
                            pre_buf.clear()
                            pre_buf_samples = 0
                    else:
                        if above:
                            silence_windows = 0
                        else:
                            silence_windows += 1

                        if silence_windows >= SILENCE_WINDOWS:
                            in_speech = False
                            silence_windows = 0
                            onset_streak = 0
                            pre_buf.clear()
                            pre_buf_samples = 0
                            if speech_samples >= MIN_SPEECH_SAMPLES:
                                captured = bytes(speech_buf)
                                speech_buf = bytearray()
                                speech_samples = 0
                                asyncio.create_task(process_turn(captured))
                            else:
                                speech_buf = bytearray()
                                speech_samples = 0
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[live] unhandled exception: {e!r}")
    except BaseException as e:
        print(f"[live] unhandled base exception: {type(e).__name__}: {e!r}")
        raise
    finally:
        for c in active_reqs:
            await c.aclose()
