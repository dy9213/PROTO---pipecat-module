# Voice Chat Demo — SPEC.md

> Standalone Electron chat app with text input, push-to-talk, and live voice chat mode.
> Connects to external STT-Whisper, LM Studio, and Kokoro-TTS services via configurable endpoints.
> English and Japanese support. Single page. Self-contained Python venv.

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [Folder structure](#2-folder-structure)
3. [Tech stack](#3-tech-stack)
4. [External service dependencies](#4-external-service-dependencies)
5. [Electron shell](#5-electron-shell)
6. [Single page UI layout](#6-single-page-ui-layout)
7. [UI modes and state machine](#7-ui-modes-and-state-machine)
8. [Space bar interaction logic](#8-space-bar-interaction-logic)
9. [Language switching](#9-language-switching)
10. [FastAPI backend](#10-fastapi-backend)
11. [Audio pipeline](#11-audio-pipeline)
12. [Live chat WebSocket protocol](#12-live-chat-websocket-protocol)
13. [Chat history](#13-chat-history)
14. [Settings persistence](#14-settings-persistence)
15. [Bootstrap and install](#15-bootstrap-and-install)
16. [Python requirements](#16-python-requirements)

---

## 1. What this is

A minimal Electron desktop app demonstrating three interaction modes in a single chat interface:

- **Text mode** — standard keyboard chat
- **PTT mode** — hold space to speak, release sends transcript as a text message
- **Live mode** — hold space 3+ seconds, enters continuous voice conversation until space is pressed again

The app does not run Whisper or Kokoro locally. It connects to separately-running services via configurable endpoint addresses. LM Studio is used as-is via OpenAI-compatible API.

---

## 2. Folder structure

```
voice-chat-demo/
├── app/
│   ├── main.js                 # Electron main process
│   ├── preload.js              # Context bridge
│   └── index.html              # Single page — all UI, vanilla JS
├── backend/
│   ├── main.py                 # FastAPI app
│   └── requirements.txt
├── data/
│   └── settings.json           # Persisted endpoint URLs and language
├── scripts/
│   ├── bootstrap.sh            # Mac/Linux venv setup
│   └── bootstrap.bat           # Windows venv setup
├── venv/                       # Python virtual env — gitignored
├── .env.example
├── .gitignore
└── package.json
```

---

## 3. Tech stack

### Frontend
- Electron (latest stable) — no framework
- Vanilla JS + CSS in a single `index.html`
- Web Audio API — microphone capture, audio playback
- Native WebSocket API — live chat mode streaming
- Native EventSource API — LLM SSE streaming

### Backend
- Python 3.11 (must be < 3.13 for kokoro compatibility)
- FastAPI + uvicorn
- httpx — async HTTP calls to Whisper and Kokoro endpoints
- python-multipart — audio file uploads
- websockets — live chat WebSocket handler
- numpy + soundfile — audio format conversion (PCM ↔ WAV)

No ML libraries in the venv. Whisper and Kokoro run as external services.

---

## 4. External service dependencies

The user runs these separately. The app only makes HTTP calls to them.

| Service | Default endpoint | Protocol |
|---|---|---|
| faster-whisper-server | `http://localhost:9000` | HTTP POST multipart audio |
| LM Studio | `http://localhost:1234/v1` | OpenAI-compatible REST + SSE |
| Kokoro-FastAPI | `http://localhost:8880/v1` | OpenAI-compatible TTS REST |

### faster-whisper-server
Any OpenAI-compatible Whisper server works. Recommended: `faster-whisper-server` or `whisper.cpp` with server mode. Endpoint called: `POST /v1/audio/transcriptions` with `multipart/form-data` audio file + `language` param.

### Kokoro-FastAPI
Exposes OpenAI-compatible TTS: `POST /v1/audio/speech` with `{"model": "kokoro", "input": "...", "voice": "...", "response_format": "wav"}`. Returns raw WAV bytes.

Language is controlled by voice selection:
- English voices: `af_heart`, `af_sky`, `am_adam` etc. (lang_code `a`)
- Japanese voices: `jf_alpha`, `jm_kumo` etc. (lang_code `j`)

---

## 5. Electron shell

### main.js responsibilities
- Check for `venv/` on launch — run bootstrap if missing (show loading overlay)
- Spawn FastAPI: `venv/bin/python -m uvicorn backend.main:app --port 8743 --host 127.0.0.1`
- Poll `http://localhost:8743/health` every 500ms, show main window when ready
- On quit: `POST http://localhost:8743/shutdown`, force-close after 2s

### preload.js — context bridge
Minimal surface:
```js
window.api = {
  getSettings: ()    => ipcRenderer.invoke('get-settings'),
  saveSettings: (s)  => ipcRenderer.invoke('save-settings', s),
}
```
All other communication is direct HTTP/WebSocket to `localhost:8743`.

### Window
- Size: 900×700, min 700×500
- Frameless: false (keep native title bar for simplicity)
- Single BrowserWindow, no navigation

---

## 6. Single page UI layout

```
┌────────────────────────────────────────────────┐
│  Endpoint config bar (collapsible, top)        │
│  [STT: ____________] [LLM: ____________]       │
│  [TTS: ____________] [EN | JP] [▶ Save]        │
├────────────────────────────────────────────────┤
│                                                │
│  Chat message list                             │
│  (scrollable, fills remaining height)         │
│                                                │
│  User bubbles — right aligned                  │
│  Assistant bubbles — left aligned             │
│                                                │
│  [Mode indicator — visible in live mode]       │
│                                                │
├────────────────────────────────────────────────┤
│  Input bar (fades in PTT/live mode)            │
│  [🎙 Hold SPACE] [text input          ] [Send] │
│  [Space hint label]                            │
└────────────────────────────────────────────────┘
```

### Endpoint config bar
- Collapsed by default — click a chevron or gear icon to expand
- Three text inputs: STT, LLM, TTS endpoints
- Language toggle: `EN` / `JP` pill buttons
- Save button — POSTs to `/settings`
- Shows green/red dot per endpoint (connection status, checked on save)

### Chat message list
- Fills available height between config bar and input bar
- Auto-scrolls to bottom on new message
- Persists across the session in memory — no reload required
- Messages have: role (user/assistant), content (text), source tag (text/voice), timestamp

### Message bubbles
- User: right-aligned, dark background
- Assistant: left-aligned, lighter background, streams in token by token
- Small mic icon on voice-originated user messages
- Typing indicator (animated dots) while LLM is generating

### Input bar
- Mic hint label: "Hold SPACE to speak · Hold 3s for live mode"
- Text input: standard, send on Enter or click Send
- In PTT active state: text input blurs, mic glow animation shows
- In live mode: entire input bar fades to 20% opacity (still visible, not gone)

### Live mode overlay
- Appears centered above input bar during live mode
- Shows: animated waveform (CSS only), "Live Chat Active", "Press SPACE to exit"
- Does not block the chat message list

---

## 7. UI modes and state machine

Five states. All transitions described below in space bar section.

```
IDLE
  │ space keydown
  ▼
PTT_RECORDING  (mic active, timer starts)
  │ space keyup (< 3s)        │ timer reaches 3s
  ▼                            ▼
PTT_PROCESSING             LIVE_ACTIVE
  │ done                       │ space keydown (tap, not hold)
  ▼                            ▼
IDLE                        IDLE
```

| State | Chat visible | Input bar | Mic indicator | Live overlay |
|---|---|---|---|---|
| IDLE | Yes | Full opacity | Off | Hidden |
| PTT_RECORDING | Yes | Dim (60%) | Glowing red | Hidden |
| PTT_PROCESSING | Yes | Dim (60%) | Spinning | Hidden |
| LIVE_ACTIVE | Yes | 20% opacity | Pulse animation | Visible |

---

## 8. Space bar interaction logic

### Constants (defined at top of script block in index.html)
```js
const LIVE_THRESHOLD_MS = 3050  // 50ms grace window past 3s prevents edge triggering
const PTT_COOLDOWN_MS   = 800   // cooldown after PTT or live session ends
```

### Guard conditions — checked on every keydown before any state change
Three guards must all pass for a space keydown to enter PTT_RECORDING:

1. `e.repeat === false` — ignore OS auto-repeat events from a held key
2. `document.activeElement !== textInput` — text box focus takes full priority, space is never intercepted while typing
3. `Date.now() >= pttCooldownUntil` — cooldown has expired since last PTT/live exit
4. `state === STATE.IDLE` — no existing session in progress

The cooldown only activates on PTT/live exit, and only applies to the PTT trigger path. The text input is unaffected — the `activeElement` guard fires before the cooldown is even read.

### Cancel token pattern — eliminates the 3-second boundary race

The `setTimeout` for live mode transition and the `keyup` handler can both fire in the same event loop tick when the user releases space near 3000ms. A single state variable alone does not prevent this — both handlers can read the same state before either writes a new value.

The fix is a cancel token: a `Symbol()` minted on each keydown. The timeout checks its token is still current before transitioning. `keyup` invalidates the token synchronously — making the timeout a guaranteed no-op regardless of event ordering.

```js
let state            = STATE.IDLE
let liveModeToken    = null
let pttCooldownUntil = 0

function isPTTEligible() {
    if (document.activeElement === textInput) return false
    if (Date.now() < pttCooldownUntil) return false
    return true
}

document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space') return
    if (e.repeat) return
    if (state !== STATE.IDLE) return
    if (!isPTTEligible()) return

    state = STATE.PTT_RECORDING
    startRecording()

    const token = Symbol()          // unique identity for this keydown
    liveModeToken = token

    setTimeout(() => {
        if (liveModeToken !== token) return     // token invalidated — no-op
        if (state !== STATE.PTT_RECORDING) return
        state = STATE.LIVE_ACTIVE
        liveModeToken = null
        enterLiveMode()
    }, LIVE_THRESHOLD_MS)
})

document.addEventListener('keyup', (e) => {
    if (e.code !== 'Space') return

    liveModeToken = null    // invalidate token — synchronous, always safe to call

    if (state === STATE.PTT_RECORDING) {
        state = STATE.PTT_PROCESSING
        pttCooldownUntil = Date.now() + PTT_COOLDOWN_MS
        processPTT()
    } else if (state === STATE.LIVE_ACTIVE) {
        state = STATE.IDLE
        pttCooldownUntil = Date.now() + PTT_COOLDOWN_MS
        exitLiveMode()
    }
    // any other state: keyup is a no-op
})
```

### processPTT — async guard
The PTT flow is async. A mid-flight guard prevents stale results from landing if state changes during the upload:

```js
async function processPTT() {
    try {
        const transcript = await uploadAudio()
        if (state !== STATE.PTT_PROCESSING) return  // aborted — discard
        appendUserMessage(transcript, 'voice')
        await streamLLMResponse(transcript)
        const audio = await fetchTTS(lastAssistantText)
        if (state !== STATE.PTT_PROCESSING) return  // aborted mid-TTS
        playAudio(audio)
    } finally {
        state = STATE.IDLE   // always return to IDLE, even on error
    }
}
```

### Cooldown behaviour
- Activates only on PTT_RECORDING → PTT_PROCESSING transition and on LIVE_ACTIVE → IDLE transition
- Duration: `PTT_COOLDOWN_MS` (800ms default)
- Scope: PTT trigger path only — text input is completely unaffected
- Effect: a tap immediately after ending a session is silently ignored, no visual error shown
- Tune by adjusting `PTT_COOLDOWN_MS` constant — 500ms minimum, 1200ms conservative

### PTT flow (hold and release under 3 seconds)

```
keydown(space) [guards pass]
  → state = PTT_RECORDING
  → mint cancel token
  → start MediaRecorder (audio/webm)
  → start 3s arc animation
  → show mic indicator (glowing red)

keyup(space) before 3050ms
  → liveModeToken = null  (timeout becomes no-op)
  → stop MediaRecorder
  → state = PTT_PROCESSING
  → start cooldown
  → processPTT() async:
      upload audio → /stt → transcript
      stream response → /chat/stream
      fetch audio → /tts
      play audio
  → state = IDLE
```

### Live mode flow (hold 3+ seconds)

```
setTimeout fires at 3050ms [token still valid]
  → state = LIVE_ACTIVE
  → discard PTT recording buffer
  → show live overlay
  → open WebSocket to /live
  → start AudioWorklet capture (16kHz mono PCM)
  → stream binary audio frames to backend

keydown(space) while LIVE_ACTIVE [e.repeat guard fires first if held]
  → state = IDLE
  → start cooldown
  → close WebSocket
  → stop AudioWorklet
  → stop audio playback
  → hide live overlay
```

### 3-second progress indicator
During PTT_RECORDING, a thin arc around the mic button animates from 0° to 360° over 3 seconds using CSS `conic-gradient` or SVG `stroke-dashoffset`. At 3050ms the arc completes and pulses once to signal mode transition. Implemented in CSS only — no canvas, no JS animation loop. The animation is reset on `keyup` if it fires before completion.

### Interrupt in live mode
If the user speaks while assistant audio is playing:
1. Frontend sends `{"type": "interrupt"}` WebSocket frame
2. Frontend immediately cancels all queued `AudioBufferSourceNode` playback
3. Backend receives interrupt, calls `request.aclose()` on in-flight LM Studio and Kokoro requests
4. Backend resets VAD buffer and resumes listening

---

## 9. Language switching

Language is a global setting stored in `data/settings.json`. Toggle in the config bar switches between `en` and `ja`.

### Effect on each service

**STT (Whisper)**
- English: `language=en` in the transcription request
- Japanese: `language=ja` in the transcription request

**LM Studio**
- No language param — the system prompt instructs the model to respond in the active language
- System prompt injected per language:
  - English: `"You are a helpful voice assistant. Respond concisely in English."`
  - Japanese: `"あなたは役立つ音声アシスタントです。日本語で簡潔に回答してください。"`

**Kokoro TTS**
- English voices (lang_code `a`): `af_heart` (female), `am_adam` (male)
- Japanese voices (lang_code `j`): `jf_alpha` (female), `jm_kumo` (male)
- Voice is selected automatically based on active language
- Backend passes the correct voice to Kokoro-FastAPI in the TTS request

### Language-aware settings stored
```json
{
  "stt_endpoint": "http://localhost:9000",
  "llm_endpoint": "http://localhost:1234/v1",
  "tts_endpoint": "http://localhost:8880/v1",
  "language": "en",
  "voice_en": "af_heart",
  "voice_ja": "jf_alpha"
}
```

---

## 10. FastAPI backend

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Readiness check |
| GET | `/settings` | Read `data/settings.json` |
| POST | `/settings` | Write settings, test endpoint connectivity |
| POST | `/stt` | Receive audio file, forward to Whisper, return transcript |
| POST | `/chat/stream` | SSE — forward message to LM Studio, stream tokens |
| POST | `/tts` | Forward text to Kokoro, return WAV audio bytes |
| WS | `/live` | WebSocket — full duplex live voice chat |
| POST | `/shutdown` | Clean exit |

### /stt
- Receives: `multipart/form-data` with `audio` file (webm) and `language` string
- Converts webm → wav using soundfile/numpy if needed
- Forwards to `{stt_endpoint}/v1/audio/transcriptions`
- Returns: `{"transcript": "..."}`

### /chat/stream
- Receives: `{"message": "...", "history": [...], "language": "en"}`
- Injects language-appropriate system prompt
- Forwards to `{llm_endpoint}/chat/completions` with `stream: true`
- Proxies SSE tokens back to caller
- Returns OpenAI-format SSE stream

### /tts
- Receives: `{"text": "...", "language": "en"}`
- Selects correct voice based on language setting
- POST to `{tts_endpoint}/audio/speech` with `{"model": "kokoro", "input": "...", "voice": "...", "response_format": "wav"}`
- Returns raw WAV bytes with `Content-Type: audio/wav`

### /settings POST
After saving, tests each endpoint:
- STT: `GET {stt_endpoint}/health` or equivalent
- LLM: `GET {llm_endpoint}/models`
- TTS: `GET {tts_endpoint}/audio/voices` or `/health`
- Returns `{"stt": true/false, "llm": true/false, "tts": true/false}`

---

## 11. Audio pipeline

### PTT recording (frontend)
- `MediaRecorder` with `audio/webm;codecs=opus`
- Collects chunks on `dataavailable` (every 250ms)
- On stop: assemble Blob, POST to `/stt` as `multipart/form-data`

### PTT playback (frontend)
- `/tts` returns WAV bytes
- Decode with `AudioContext.decodeAudioData()`
- Play via `AudioContext.createBufferSource()`

### Live mode audio streaming (frontend → backend)
- `AudioWorklet` or `ScriptProcessor` captures raw PCM at 16kHz mono
- Sends 20ms chunks (320 samples) as binary WebSocket frames
- Receives back: JSON text events (transcript, tokens) and binary audio chunks

### Live mode audio playback (frontend)
- Receives WAV chunks from backend over WebSocket
- Queues into `AudioContext` for gapless playback
- On `interrupt` message: immediately cancel queued buffers and current source

### Backend audio handling
- Receives raw PCM 16kHz mono Int16 from WebSocket
- Accumulates into a ring buffer
- Simple energy-based VAD: detect speech end when RMS drops below threshold for 600ms
- On speech end: encode buffer as WAV → POST to Whisper → get transcript → process turn

---

## 12. Live chat WebSocket protocol

All JSON frames are text. Binary frames are raw audio (PCM Int16 from client, WAV chunks from server).

### Client → Server

```json
{"type": "audio_config", "sample_rate": 16000, "channels": 1}
{"type": "interrupt"}
{"type": "ping"}
```

Binary: raw PCM Int16 audio chunks (20ms = 320 samples at 16kHz)

### Server → Client

```json
{"type": "transcript",   "text": "what's the weather like?", "final": true}
{"type": "token",        "text": " The", "agent": "assistant"}
{"type": "tts_start"}
{"type": "tts_end"}
{"type": "interrupted"}
{"type": "error",        "message": "STT endpoint unreachable"}
{"type": "pong"}
```

Binary: WAV audio chunks (streamed TTS output, 24kHz)

### Turn flow in live mode
```
Client sends PCM audio chunks...
  → VAD detects end of speech
  → Server emits {"type": "transcript", "text": "...", "final": true}
  → Server calls LM Studio, streams tokens back as {"type": "token"} events
  → Server calls Kokoro TTS, streams WAV chunks back as binary frames
  → Server emits {"type": "tts_start"} before first audio chunk
  → Server emits {"type": "tts_end"} after last audio chunk
  → Client resumes listening for next speech
```

---

## 13. Chat history

Stored in memory only (no SQLite for this demo). Cleared on app restart.

Each message:
```js
{
  id: uuid,
  role: "user" | "assistant",
  content: "...",
  source: "text" | "voice",   // shows mic icon if voice
  timestamp: Date.now(),
  streaming: false            // true while assistant is generating
}
```

History is sent with each `/chat/stream` request as `history` array (last 10 messages only, to limit context size).

---

## 14. Settings persistence

Stored in `data/settings.json`. Read on backend startup. Written via `POST /settings`.

Defaults:
```json
{
  "stt_endpoint": "http://localhost:9000",
  "llm_endpoint": "http://localhost:1234/v1",
  "tts_endpoint": "http://localhost:8880/v1",
  "language": "en",
  "voice_en": "af_heart",
  "voice_ja": "jf_alpha"
}
```

Frontend reads settings on load via `GET /settings` and populates the endpoint config bar.

---

## 15. Bootstrap and install

### scripts/bootstrap.sh
```bash
#!/bin/bash
set -e
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
mkdir -p data
cp .env.example .env 2>/dev/null || true
echo "Bootstrap complete. Run: npm start"
```

### scripts/bootstrap.bat
```bat
@echo off
python -m venv venv
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r backend\requirements.txt
if not exist data mkdir data
echo Bootstrap complete. Run: npm start
```

### package.json scripts
```json
{
  "scripts": {
    "start":     "electron app/main.js",
    "bootstrap": "bash scripts/bootstrap.sh",
    "dev":       "NODE_ENV=development electron app/main.js"
  }
}
```

### Electron launch sequence
1. Check `venv/` exists — if not, show loading screen and run bootstrap
2. Spawn FastAPI on port 8743
3. Poll `/health` — show window when ready (max 20s timeout)
4. Load `index.html`
5. On quit: POST `/shutdown`, wait 2s, `app.quit()`

---

## 16. Python requirements

`backend/requirements.txt`:
```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0
python-multipart>=0.0.9
websockets>=12.0
numpy>=1.26.0
soundfile>=0.12.1
```

No ML libraries. No torch. No whisper. No kokoro.
All model inference is delegated to external services.
Total venv install time: ~30 seconds.
Total venv size: ~150MB (mostly numpy).

### System dependencies (user must have installed)
- Node.js 18+ (for Electron)
- Python 3.11 (must be < 3.13)
- On Linux: `libsndfile1` (for soundfile)
- On Windows: no extras needed
- On macOS: no extras needed

### External services the user runs separately
- `faster-whisper-server` — e.g. via pip or Docker
- `LM Studio` — already running
- `Kokoro-FastAPI` — e.g. via Docker (`ghcr.io/remsky/kokoro-fastapi-cpu:latest`)

---

## Notes for Claude Code

- Vanilla JS only in `index.html` — no React, no bundler, no node_modules in `app/`
- All CSS inline or in a `<style>` block in `index.html` — single file
- The space bar state machine is the most critical piece — implement exactly as specified in section 8 using the cancel token pattern. Do NOT use boolean flags or a plain state variable alone — both have race conditions at the 3-second boundary
- Constants `LIVE_THRESHOLD_MS` and `PTT_COOLDOWN_MS` must be defined at the top of the script block, not inline — they will need tuning
- The `e.repeat` guard on `keydown` is mandatory — without it, the OS fires repeated keydown events while space is held and the state machine breaks
- Cooldown is scoped to PTT exit only and never affects the text input — the `activeElement` check fires before the cooldown is read
- `MediaRecorder` in Electron works identically to browser — no special handling needed
- `AudioWorklet` required for live audio capture — `ScriptProcessor` is deprecated and unreliable in Electron
- For the 3-second arc animation — pure CSS `conic-gradient` or SVG `stroke-dashoffset`, no canvas, no JS animation loop
- Backend `/live` WebSocket handler must handle interrupt gracefully: `request.aclose()` on in-flight httpx requests to LM Studio and Kokoro, discard VAD buffer, emit `{"type": "interrupted"}` before resuming listen
- Energy-based RMS VAD is sufficient — no silero-vad, no torch
- `VAD_RMS_THRESHOLD = 0.01` and `VAD_SILENCE_MS = 600` defined as constants at top of `main.py`
- `processPTT()` must have the mid-flight state guard and `finally: state = STATE.IDLE` — without it, errors leave the app stuck in PTT_PROCESSING
- Keep backend under 200 lines — thin proxy only

---

*Spec version: 1.1 — cancel token pattern + cooldown added to section 8*
