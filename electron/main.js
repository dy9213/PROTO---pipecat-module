const { app, BrowserWindow, ipcMain } = require('electron')
const path   = require('path')
const fs     = require('fs')
const http   = require('http')
const { spawn, execSync } = require('child_process')

const ROOT          = path.join(__dirname, '..')
const VENV_PYTHON   = path.join(ROOT, 'venv', 'bin', 'python')
const SETTINGS_PATH = path.join(ROOT, 'data', 'settings.json')
const BACKEND_PORT  = 8743
const HEALTH_URL    = `http://127.0.0.1:${BACKEND_PORT}/health`

let mainWindow  = null
let backendProc = null

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
  })
  backendProc.stdout.on('data', (d) => process.stdout.write(d))
  backendProc.stderr.on('data', (d) => process.stderr.write(d))
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
  mainWindow.loadFile(path.join(ROOT, 'app', 'index.html'))
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
  try {
    await fetch(`http://127.0.0.1:${BACKEND_PORT}/shutdown`, { method: 'POST' })
  } catch {}
  await new Promise((r) => setTimeout(r, 2000))
  backendProc.kill()
})
