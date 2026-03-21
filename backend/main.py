import asyncio, json, os, sys, tempfile, urllib.request
from pathlib import Path

# ── production path roots (set by Electron, fall back to dev layout) ──────────
APP_ROOT  = Path(os.environ.get("ONICHAT_APP_ROOT",  Path(__file__).parent.parent))
USER_DATA = Path(os.environ.get("ONICHAT_USER_DATA", Path(__file__).parent.parent))
from typing import AsyncGenerator

# Add project root to sys.path so `modules/` package is importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from modules.tts.tts_manager import voicevox_tts as _voicevox_tts
from modules.stt.stt_manager import stt_manager, MODELS as STT_MODELS, AVAILABLE as STT_AVAILABLE, DEFAULT_MODEL as STT_DEFAULT_MODEL
from modules.llm.llm_manager import llm_manager, MODEL_FILES as LLM_MODEL_FILES, LLAMA_URL as LLM_LOCAL_URL
from modules.tts.voicevox_manager import voicevox_manager

# ── debug ──────────────────────────────────────────────────────────────────────
DEBUG_RAW_SSE = False   # set True to print every raw SSE line from the LLM
DEBUG_VAD     = False   # set True to print VAD prob + silence window counts
DEBUG_SEARCH  = True    # set True to print web search query + results

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

SETTINGS_PATH = USER_DATA / "data" / "settings.json"
DEFAULT_SETTINGS = {
    "stt_endpoint":      "",
    "stt_model":         "qwen3-1.7b-4bit",
    "llm_endpoint":      "",
    "tts_endpoint":      "",
    "tts_mode":          "voicevox",
    "voicevox_speaker":  2,
    "llm_model":         "",
    "llm_api_key":       "",
    "language":          "ja",
    "voice_en":          "Ryan",
    "voice_ja":          "Ono_Anna",
    "voice_zh":          "Vivian",
    "system_prompt_en":  "",
    "system_prompt_ja":  "あなたは日本語の会話相手です。返信は短くしましょう。出力はテキスト読み上げ用です。",
    "system_prompt_zh":  "",
    "search_online":     False,
    "translate_to":      "en",
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
    import datetime
    lang = s.get("language", "en")
    key  = f"system_prompt_{lang}"
    sp   = s.get(key, "").strip()
    base = sp if sp else DEFAULT_SYSTEM_PROMPTS.get(lang, DEFAULT_SYSTEM_PROMPTS["en"])
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    label = {"en": "Current date and time", "ja": "現在の日時", "zh": "当前日期和时间"}.get(lang, "Current date and time")
    return f"{base}\n\n{label}: {now}"


def _is_local_llm(s: dict) -> bool:
    return s.get("llm_model", "") in _LOCAL_LLM_KEYS


def _llm_extra(s: dict) -> dict:
    """Extra JSON fields for the chat completions request.
    Disables Qwen3 thinking mode on local models via chat_template_kwargs."""
    if _is_local_llm(s):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}

# ── LLM / TTS helpers ─────────────────────────────────────────────────────────
_LOCAL_LLM_KEYS = frozenset(LLM_MODEL_FILES.keys())

def _llm_config(s: dict) -> tuple[str, dict, str]:
    """Return (endpoint_base, auth_headers, model_name) for the current LLM setting."""
    key = s.get("llm_model", "")
    if key in _LOCAL_LLM_KEYS:
        return LLM_LOCAL_URL, {}, "local-model"
    endpoint = s.get("llm_endpoint", "").rstrip("/")
    headers  = {"Authorization": f"Bearer {s['llm_api_key']}"} if s.get("llm_api_key") else {}
    model    = s.get("llm_model") or "local-model"
    return endpoint, headers, model


async def _ensure_llm_running(s: dict) -> None:
    """Start llama-server if a local model key is selected but not yet running."""
    key = s.get("llm_model", "")
    if key not in _LOCAL_LLM_KEYS:
        return
    if llm_manager.is_running() and llm_manager._active_key == key:
        return
    if not llm_manager.is_model_present(key):
        raise RuntimeError(
            f"Local model '{key}' is not downloaded — open the model selector to download it first."
        )
    print(f"[llm] starting llama-server for '{key}'…", flush=True)
    await asyncio.to_thread(llm_manager.start, key)
    print("[llm] llama-server ready", flush=True)

def _voicevox_endpoint(s: dict) -> str:
    """Return the active VOICEVOX endpoint (always local when tts_mode is voicevox)."""
    if voicevox_manager.is_running():
        return voicevox_manager.endpoint
    # Never fall back to the remote tts_endpoint when voicevox mode is selected —
    # caller should have ensured the engine is running via _ensure_voicevox_running.
    if s.get("tts_mode") == "voicevox":
        return voicevox_manager.endpoint   # will fail at connect time with a clear error
    return s.get("tts_endpoint", "").rstrip("/")


async def _ensure_voicevox_running(s: dict) -> None:
    """Start voicevox_engine if it's the selected TTS mode and not yet running."""
    if s.get("tts_mode") != "voicevox":
        return
    if voicevox_manager.is_running():
        return
    if not voicevox_manager.is_installed():
        raise RuntimeError("VOICEVOX engine is not installed — open the launcher to download it.")
    print("[tts] starting voicevox_engine…", flush=True)
    await asyncio.to_thread(voicevox_manager.start)
    print("[tts] voicevox_engine ready", flush=True)

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def _autoload_services():
    """Placeholder — all services (LLM, VOICEVOX, STT) are started on Launch button click."""
    pass

@app.on_event("shutdown")
async def _shutdown_services():
    llm_manager.stop()
    voicevox_manager.stop()

def load_settings() -> dict:
    try:
        s = {**DEFAULT_SETTINGS, **json.loads(SETTINGS_PATH.read_text())}
    except Exception:
        s = DEFAULT_SETTINGS.copy()
    # normalise legacy "remote" value
    if s.get("tts_mode") == "remote":
        s["tts_mode"] = "kokoro"
    return s

def save_settings(s: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2))

# ── helpers ────────────────────────────────────────────────────────────────────
def webm_to_wav(data: bytes) -> bytes:
    """Decode webm/any audio bytes → 16 kHz mono WAV bytes via soundfile + scipy."""
    import io
    from scipy.signal import resample_poly
    from math import gcd
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as src:
        src.write(data); src_path = src.name
    try:
        audio, sr = sf.read(src_path, always_2d=True)
        # Mix down to mono
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        audio = audio[:, 0]
        # Resample to 16 kHz if needed
        target_sr = 16000
        if sr != target_sr:
            g = gcd(target_sr, sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype("float32")
        buf = io.BytesIO()
        sf.write(buf, audio, target_sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    finally:
        os.unlink(src_path)

def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, samples, sample_rate, subtype="PCM_16")
        return Path(f.name).read_bytes()

# ── web search ─────────────────────────────────────────────────────────────────
_REFORMULATE_PROMPT = {
    "en": "Today is {date}. Convert the user message into a concise web search query. Output only the search query, no explanation, no punctuation.",
    "ja": "今日は{date}です。ユーザーのメッセージを簡潔な日本語のウェブ検索クエリに変換してください。必ず日本語で出力し、説明や句読点は不要です。",
    "zh": "今天是{date}。将用户消息转换为简洁的网络搜索查询。只输出搜索查询，不需要解释或标点符号。",
}

async def _reformulate_query(endpoint: str, headers: dict, model: str, message: str,
                             language: str = "ja", extra: dict | None = None) -> str:
    """Quick non-streaming LLM call to turn a conversational message into a clean search query."""
    import datetime
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = _REFORMULATE_PROMPT.get(language, _REFORMULATE_PROMPT["en"]).format(date=date)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=5)) as c:
            r = await c.post(
                endpoint + "/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": message},
                    ],
                    "stream": False,
                    "max_tokens": 30,
                    **(extra or {}),
                },
            )
            r.raise_for_status()
            query = r.json()["choices"][0]["message"]["content"].strip()
            return query or message
    except Exception as e:
        print(f"[search] reformulation failed: {e}", flush=True)
        return message

def search_web(query: str, max_results: int = 5) -> str:
    """Synchronous DDG search — call via asyncio.to_thread."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results, region="jp-jp")
        if not results:
            return ""
        parts = [f"{r['title']}\n{r['href']}\n{r['body']}" for r in results]
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[search] error: {e}", flush=True)
        return ""

# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/system/info")
async def system_info():
    from modules.system.system_info import get_system_info
    return await asyncio.to_thread(get_system_info)

@app.get("/system/services")
async def services_status():
    """Health check for all local services — binary installed + process running."""
    from modules.llm.installer import is_installed as llama_installed
    from modules.tts.installer  import is_installed as voicevox_installed
    s = load_settings()
    return {
        "llama_server":    {"installed": llama_installed(),         "running": llm_manager.is_running(),         "model": llm_manager._active_key},
        "voicevox_engine": {"installed": voicevox_installed(),      "running": voicevox_manager.is_running()},
        "stt":             {"loaded": stt_manager.loaded_model is not None, "model": stt_manager.active_key},
    }

@app.get("/system/install-check")
async def install_check():
    from modules.llm.installer import is_installed as llama_installed
    from modules.tts.installer  import is_installed as voicevox_installed
    return {
        "llama_server":    llama_installed(),
        "voicevox_engine": voicevox_installed(),
    }

def _sse_install_stream(install_fn):
    """Shared SSE helper: runs install_fn(cb) in a thread, streams progress."""
    q    = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def cb(pct: int, msg: str):
        loop.call_soon_threadsafe(q.put_nowait, {"pct": pct, "msg": msg})

    async def run():
        try:
            await asyncio.to_thread(install_fn, cb)
        except Exception as e:
            await q.put({"error": str(e)})

    task = asyncio.create_task(run())

    async def stream():
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("pct") == 100 or "error" in item:
                    break
            except asyncio.TimeoutError:
                if task.done():
                    break
                yield ": keep-alive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/system/install")
async def system_install_llama():
    from modules.llm.installer import install
    return _sse_install_stream(install)

@app.get("/system/voicevox-install")
async def system_install_voicevox():
    from modules.tts.installer import install
    return _sse_install_stream(install)

@app.get("/llm/status")
async def llm_status():
    return {"running": llm_manager.is_running(), "model": llm_manager._active_key}

@app.get("/llm/models")
async def llm_models():
    return llm_manager.models_status()

@app.post("/llm/start/{key}")
async def llm_start(key: str):
    from fastapi import HTTPException
    if key not in _LOCAL_LLM_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {key}")
    if not llm_manager.is_model_present(key):
        raise HTTPException(status_code=400, detail=f"Model '{key}' not downloaded")
    try:
        await asyncio.to_thread(llm_manager.start, key)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/llm/stop")
async def llm_stop():
    llm_manager.stop()
    return {"ok": True}

@app.get("/llm/download/{key}")
async def llm_download(key: str):
    if key not in _LOCAL_LLM_KEYS:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown model key: {key}")

    q    = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def cb(pct: int, msg: str):
        loop.call_soon_threadsafe(q.put_nowait, {"pct": pct, "msg": msg})

    async def run():
        try:
            await asyncio.to_thread(llm_manager.download_model, key, cb)
        except Exception as e:
            await q.put({"error": str(e)})

    task = asyncio.create_task(run())

    async def stream():
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=60)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("pct") == 100 or "error" in item:
                    break
            except asyncio.TimeoutError:
                if task.done():
                    break
                yield ": keep-alive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/voicevox/status")
async def voicevox_status():
    return {
        "running":   voicevox_manager.is_running(),
        "installed": voicevox_manager.is_installed(),
    }

@app.post("/voicevox/start")
async def voicevox_start():
    from fastapi import HTTPException
    if not voicevox_manager.is_installed():
        raise HTTPException(status_code=400, detail="VOICEVOX engine is not installed")
    if voicevox_manager.is_running():
        return {"ok": True, "already_running": True}
    try:
        await asyncio.to_thread(voicevox_manager.start)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voicevox/warmup")
async def voicevox_warmup():
    """Synthesise a short silent phrase to force speaker 2 weights into RAM."""
    s = load_settings()
    speaker = int(s.get("voicevox_speaker") or 2)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            q = await client.post(f"{voicevox_manager.endpoint}/audio_query",
                                  params={"text": "。", "speaker": speaker})
            if q.status_code == 200:
                await client.post(f"{voicevox_manager.endpoint}/synthesis",
                                  params={"speaker": speaker},
                                  content=q.content,
                                  headers={"Content-Type": "application/json"})
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/settings")
async def get_settings():
    return load_settings()

class SettingsIn(BaseModel):
    stt_endpoint:     str = ""
    stt_model:        str = "qwen3-1.7b-4bit"
    llm_endpoint:     str = ""
    tts_endpoint:     str = ""
    tts_mode:         str = "kokoro"
    voicevox_speaker: int = 2
    llm_model:        str = ""
    llm_api_key:      str = ""
    language:         str = "ja"
    voice_en:         str = "Ryan"
    voice_ja:         str = "Ono_Anna"
    voice_zh:         str = "Vivian"
    system_prompt_en: str = ""
    system_prompt_ja: str = ""
    system_prompt_zh: str = ""
    search_online:    bool = False
    translate_to:     str  = "en"

@app.post("/settings")
async def post_settings(body: SettingsIn):
    s_old = load_settings()
    s = body.model_dump()
    save_settings(s)
    if s.get("stt_model") != s_old.get("stt_model"):
        stt_manager.unload()
        stt_manager.active_key = s.get("stt_model", STT_DEFAULT_MODEL)
    tts_health_path = "/speakers" if s.get("tts_mode") == "voicevox" else "/health"
    status = {"stt": False, "llm": False, "tts": False}
    async with httpx.AsyncClient(timeout=3) as c:
        for key, url, path in [
            ("stt", s["stt_endpoint"], "/health"),
            ("llm", s["llm_endpoint"], "/v1/models"),
            ("tts", s["tts_endpoint"], tts_health_path),
        ]:
            if not url: continue
            try:
                r = await c.get(url.rstrip("/") + path)
                status[key] = r.status_code < 400
            except Exception:
                pass
    return status

def _stt_model_present(repo: str | None) -> bool:
    """Check whether an mlx-community HF repo is in the local HF cache."""
    if not repo:
        return True   # remote sentinel — no local files needed
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    name  = "models--" + repo.replace("/", "--")
    return (cache / name / "snapshots").exists()

@app.get("/stt/models")
async def get_stt_models():
    return [
        {
            "key":       key,
            "label":     meta["label"],
            "backend":   meta["backend"],
            "present":   _stt_model_present(meta.get("repo")),
            "loaded":    stt_manager.active_key == key and stt_manager.loaded_model is not None,
            "available": meta["backend"] == "remote" or STT_AVAILABLE.get(meta["backend"], False),
        }
        for key, meta in STT_MODELS.items()
    ]

@app.get("/stt/download/{key}")
async def stt_download(key: str):
    import threading
    meta = STT_MODELS.get(key)
    if not meta or meta["backend"] == "remote":
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown or non-local STT key: {key}")

    repo = meta["repo"]

    def download_stt(cb):
        cb(0, "Connecting to Hugging Face…")
        done  = threading.Event()
        error = []

        def run():
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo)
            except Exception as e:
                error.append(e)
            finally:
                done.set()

        threading.Thread(target=run, daemon=True).start()
        pct = 5
        while not done.wait(timeout=2):
            cb(min(pct, 90), "Downloading from Hugging Face…")
            pct = min(pct + 3, 90)
        if error:
            raise error[0]
        cb(100, f"Ready: {key}")

    return _sse_install_stream(download_stt)

class SttLoadIn(BaseModel):
    model: str

@app.post("/stt/load")
async def post_stt_load(body: SttLoadIn):
    try:
        result = await asyncio.to_thread(stt_manager.load, body.model)
        return {"ok": True, **result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/stt/load/cancel")
async def post_stt_load_cancel():
    stt_manager.cancel_load()
    return {"cancelled": True}

@app.post("/stt")
async def stt(audio: UploadFile = File(...), language: str = Form("en")):
    s = load_settings()
    raw = await audio.read()
    is_wav = (audio.content_type == "audio/wav" or (audio.filename or "").endswith(".wav"))
    wav = raw if is_wav else webm_to_wav(raw)
    stt_model = s.get("stt_model", STT_DEFAULT_MODEL)
    if stt_model != "remote":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            tmp_path = f.name
        try:
            text = await asyncio.to_thread(stt_manager.transcribe, tmp_path, language)
        finally:
            os.unlink(tmp_path)
        return {"transcript": text}
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
    await _ensure_llm_running(s)
    messages = [{"role": "system", "content": get_system_prompt(s)}]
    messages += body.history[-10:]
    _ep, _hdrs, _mdl = _llm_config(s)
    if s.get("search_online"):
        query = await _reformulate_query(_ep, _hdrs, _mdl, body.message, s.get("language", "ja"), extra=_llm_extra(s))
        results = await asyncio.to_thread(search_web, query)
        if DEBUG_SEARCH:
            print(f"[search] original: {body.message!r}\n[search] query:    {query!r}\n[search] results:\n{results}\n", flush=True)
        if results:
            messages.append({"role": "system", "content": f"Web search results:\n\n{results}"})
    messages.append({"role": "user", "content": body.message})

    async def stream() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=5)) as c:
                async with c.stream("POST", _ep + "/v1/chat/completions",
                                     headers=_hdrs,
                                     json={"model": _mdl, "messages": messages, "stream": True,
                                           **_llm_extra(s)}) as r:
                    if r.status_code >= 400:
                        body = await r.aread()
                        yield f"data: {{\"error\": \"LLM returned {r.status_code}: {body.decode()[:200]}\"}}\n\n".encode()
                        return
                    async for line in r.aiter_lines():
                        if DEBUG_RAW_SSE:
                            print(f"[sse/chat] {line}", flush=True)
                        if line.startswith("data: "):
                            yield (line + "\n\n").encode()
        except Exception as e:
            print(f"[chat] stream error: {e}", flush=True)
            yield f"data: {{\"error\": \"{str(e)[:200]}\"}}\n\n".encode()

    return StreamingResponse(stream(), media_type="text/event-stream")

class TTSIn(BaseModel):
    text: str
    language: str = "en"

@app.post("/tts")
async def tts(body: TTSIn):
    s = load_settings()
    await _ensure_voicevox_running(s)
    voice    = s.get(f"voice_{body.language}", s["voice_en"])
    tts_lang = TTS_LANGUAGE_MAP.get(body.language, "English")

    async with httpx.AsyncClient(timeout=30) as c:
        if s.get("tts_mode") == "voicevox":
            speaker = int(s.get("voicevox_speaker") or 2)
            wav = await _voicevox_tts(c, body.text, speaker, _voicevox_endpoint(s))
        else:  # kokoro / OpenAI-compat
            r = await c.post(
                s["tts_endpoint"].rstrip("/") + "/v1/audio/speech",
                json={"model": "kokoro", "input": body.text, "voice": voice, "language": tts_lang, "response_format": "wav"},
            )
            r.raise_for_status()
            wav = r.content
    return Response(content=wav, media_type="audio/wav")

class SearchOnlineIn(BaseModel):
    enabled: bool

@app.post("/settings/search_online")
async def set_search_online(body: SearchOnlineIn):
    s = load_settings()
    s["search_online"] = body.enabled
    save_settings(s)
    return {"ok": True}

_TRANSLATE_LANG_NAMES = {"en": "English", "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish"}

class TranslateIn(BaseModel):
    text: str
    target: str | None = None  # overrides settings translate_to when provided

@app.post("/translate")
async def translate_text(body: TranslateIn):
    s = load_settings()
    await _ensure_llm_running(s)
    target = body.target or s.get("translate_to", "en")
    target_name = _TRANSLATE_LANG_NAMES.get(target, target)
    system = f"Translate the following to {target_name}. Output only the translation, no explanation."
    _ep, _hdrs, _mdl = _llm_config(s)

    async def stream() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=5)) as c:
            async with c.stream("POST", _ep + "/v1/chat/completions",
                                 headers=_hdrs,
                                 json={"model": _mdl, "stream": True,
                                       "messages": [{"role": "system", "content": system},
                                                    {"role": "user",   "content": body.text}],
                                       **_llm_extra(s)}) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        yield (line + "\n\n").encode()

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/shutdown")
async def shutdown():
    # Stop subprocesses before exiting — os._exit() bypasses the FastAPI
    # shutdown event so we must do cleanup here explicitly.
    stt_manager.unload()
    llm_manager.stop()
    voicevox_manager.stop()
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
        client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=5))
        active_reqs.append(client)
        try:
            await _ensure_llm_running(s)
            await _ensure_voicevox_running(s)
            wav = pcm_to_wav(pcm)
            stt_model = s.get("stt_model", STT_DEFAULT_MODEL)
            if stt_model != "remote":
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav)
                    tmp_path = f.name
                try:
                    transcript = await asyncio.to_thread(stt_manager.transcribe, tmp_path, s["language"])
                finally:
                    os.unlink(tmp_path)
                transcript = transcript.strip()
            else:
                r = await client.post(
                    s["stt_endpoint"].rstrip("/") + "/v1/audio/transcriptions",
                    files={"file": ("audio.wav", wav, "audio/wav")},
                    data={"model": "whisper-1", "language": s["language"]},
                )
                r.raise_for_status()
                transcript = r.json().get("text", "").strip()
            if not transcript:
                return
            # If the interrupt handler fired while the STT thread was running it
            # will have already called aclose() on client and cleared active_reqs.
            # Return silently so the stale transcript is never sent to the frontend.
            if client not in active_reqs:
                return
            await send_json({"type": "transcript", "text": transcript, "final": True})

            conv_messages.append({"role": "user", "content": transcript})
            messages = [{"role": "system", "content": get_system_prompt(s)}]
            _ep, _hdrs, _mdl = _llm_config(s)
            if s.get("search_online"):
                query = await _reformulate_query(_ep, _hdrs, _mdl, transcript, s.get("language", "ja"), extra=_llm_extra(s))
                # _reformulate_query uses its own internal client not tracked by
                # active_reqs, so it cannot be cancelled mid-flight.  Discard
                # the result silently if an interrupt arrived while it was running.
                if client not in active_reqs:
                    return
                results = await asyncio.to_thread(search_web, query)
                # search_web runs in a thread and cannot be cancelled — discard
                # its result silently if an interrupt fired while it was running.
                if client not in active_reqs:
                    return
                if DEBUG_SEARCH:
                    print(f"[search] original: {transcript!r}\n[search] query:    {query!r}\n[search] results:\n{results}\n", flush=True)
                if results:
                    messages.append({"role": "system", "content": f"Web search results:\n\n{results}"})
            messages += conv_messages[-20:]

            full_text = ""
            async with client.stream("POST", _ep + "/v1/chat/completions",
                                     headers=_hdrs,
                                     json={"model": _mdl, "stream": True,
                                           "messages": messages, **_llm_extra(s)}) as resp:
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
                voice    = s.get(f"voice_{s['language']}", s["voice_en"])
                tts_lang = TTS_LANGUAGE_MAP.get(s["language"], "English")
                await send_json({"type": "tts_start", "sample_rate": 24000})

                if s.get("tts_mode") == "voicevox":
                    speaker = int(s.get("voicevox_speaker") or 2)
                    wav = await _voicevox_tts(client, full_text, speaker, _voicevox_endpoint(s))
                    await ws.send_bytes(wav)
                else:  # kokoro / OpenAI-compat
                    r2 = await client.post(
                        s["tts_endpoint"].rstrip("/") + "/v1/audio/speech",
                        json={"model": "kokoro", "input": full_text, "voice": voice, "language": tts_lang, "response_format": "wav"},
                    )
                    r2.raise_for_status()
                    await ws.send_bytes(r2.content)

                await send_json({"type": "tts_end"})
        except Exception as e:
            # Only forward genuine errors to the frontend.
            # If the interrupt handler already removed this client from active_reqs
            # the exception was caused by our own aclose() call — swallow it silently
            # so the frontend never sees {"type": "error"} from an interrupt.
            if client in active_reqs:
                await send_json({"type": "error", "message": str(e)})
        finally:
            processing = False
            try:
                active_reqs.remove(client)
            except ValueError:
                pass  # interrupt handler already removed it
            try:
                await client.aclose()
            except Exception:
                pass  # already closed by interrupt handler

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
