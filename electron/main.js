const { app, BrowserWindow, ipcMain } = require('electron')
const path   = require('path')
const fs     = require('fs')
const http   = require('http')
const { spawn } = require('child_process')

const ROOT          = path.join(__dirname, '..')
const VENV_PYTHON   = path.join(ROOT, 'venv', 'bin', 'python')
const SETTINGS_PATH = path.join(ROOT, 'data', 'settings.json')
const BACKEND_PORT  = 8743
const HEALTH_URL    = `http://127.0.0.1:${BACKEND_PORT}/health`

let mainWindow  = null
let backendProc = null

// ── log ring buffer ────────────────────────────────────────────────────────────
const LOG_MAX   = 500
const logBuffer = []   // [{level, text}]

function pushLog(level, raw) {
  String(raw).split('\n').forEach(line => {
    if (!line) return
    const entry = { level, text: line }
    logBuffer.push(entry)
    if (logBuffer.length > LOG_MAX) logBuffer.shift()
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('log', entry)
    }
  })
}

ipcMain.handle('get-log-history', () => [...logBuffer])

// ── settings IPC ──────────────────────────────────────────────────────────────
ipcMain.handle('get-settings', () => {
  try { return JSON.parse(fs.readFileSync(SETTINGS_PATH, 'utf8')) }
  catch { return {} }
})

ipcMain.handle('save-settings', (_, s) => {
  fs.mkdirSync(path.dirname(SETTINGS_PATH), { recursive: true })
  fs.writeFileSync(SETTINGS_PATH, JSON.stringify(s, null, 2))
})

// ── health poll ───────────────────────────────────────────────────────────────
function pollHealth(resolve, reject, attempts = 0) {
  if (attempts > 240) return reject(new Error('Backend did not start in time'))
  http.get(HEALTH_URL, (res) => {
    if (res.statusCode === 200) resolve()
    else setTimeout(() => pollHealth(resolve, reject, attempts + 1), 500)
  }).on('error', () => setTimeout(() => pollHealth(resolve, reject, attempts + 1), 500))
}

// ── bootstrap ─────────────────────────────────────────────────────────────────
function runBootstrap() {
  return new Promise((resolve, reject) => {
    const proc = spawn('bash', [path.join(ROOT, 'scripts', 'bootstrap.sh')], {
      cwd: ROOT, stdio: 'inherit',
    })
    proc.on('close', (code) => code === 0 ? resolve() : reject(new Error(`Bootstrap failed (${code})`)))
  })
}

// ── spawn backend ─────────────────────────────────────────────────────────────
function startBackend() {
  backendProc = spawn(VENV_PYTHON, ['-u', '-m', 'uvicorn', 'backend.main:app',
    '--port', String(BACKEND_PORT), '--host', '127.0.0.1'], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,   // new process group (PGID = PID) — lets us kill the whole tree
  })
  backendProc.stdout.on('data', (d) => { process.stdout.write(d); pushLog('stdout', d) })
  backendProc.stderr.on('data', (d) => { process.stderr.write(d); pushLog('stderr', d) })
  backendProc.on('exit', (code) => {
    if (code !== 0 && code !== null) console.error(`Backend exited with code ${code}`)
  })
}

// ── create window ─────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 900, height: 700, minWidth: 700, minHeight: 500,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })
  mainWindow.loadFile(path.join(ROOT, 'app', 'loader.html'))
  mainWindow.on('closed', () => { mainWindow = null })
}

// ── app lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  if (!fs.existsSync(VENV_PYTHON)) {
    mainWindow = new BrowserWindow({ width: 500, height: 200, resizable: false,
      webPreferences: { contextIsolation: true } })
    mainWindow.loadURL(`data:text/html,<body style="font:16px system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#111;color:#fff">
      <div>Setting up Python environment…</div></body>`)
    try {
      await runBootstrap()
    } catch (err) {
      mainWindow.loadURL(`data:text/html,<body style="font:16px system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#111;color:#f55">
        <div>Bootstrap failed: ${err.message}</div></body>`)
      return
    }
    mainWindow.close()
    mainWindow = null
  }

  startBackend()

  await new Promise((resolve, reject) => pollHealth(resolve, reject))

  createWindow()
})

app.on('window-all-closed', () => app.quit())

app.on('before-quit', async () => {
  if (!backendProc) return
  // 1. Ask the backend to flush cleanly (unload models, close sockets).
  try {
    await fetch(`http://127.0.0.1:${BACKEND_PORT}/shutdown`, { method: 'POST' })
  } catch {}
  // 2. Give it a moment, then kill the entire process group — uvicorn, llama-server,
  //    and voicevox all share the same PGID so one signal takes down the whole tree.
  await new Promise((r) => setTimeout(r, 1000))
  try { process.kill(-backendProc.pid, 'SIGTERM') } catch {}
})
