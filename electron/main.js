const { app, BrowserWindow, ipcMain } = require('electron')
const path   = require('path')
const fs     = require('fs')
const http   = require('http')
const { spawn } = require('child_process')

// ── path roots ────────────────────────────────────────────────────────────────
// APP_ROOT  — read-only bundle root (source code, scripts)
// USER_DATA — writable runtime dir (venv, settings, downloaded binaries)
//
// Dev:  both point to the project root
// Prod: APP_ROOT = .app/Contents/Resources/app, USER_DATA = ~/Library/Application Support/OniChat
const APP_ROOT  = app.isPackaged ? path.join(process.resourcesPath, 'app') : path.join(__dirname, '..')
const USER_DATA = app.isPackaged ? app.getPath('userData') : path.join(__dirname, '..')

const VENV_PYTHON   = path.join(USER_DATA, 'venv', 'bin', 'python')
const SETTINGS_PATH = path.join(USER_DATA, 'data', 'settings.json')
const BACKEND_PORT  = app.isPackaged ? 8744 : 8743
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

ipcMain.handle('get-backend-port',   () => BACKEND_PORT)
ipcMain.on('get-backend-port-sync', (e) => { e.returnValue = BACKEND_PORT })
ipcMain.handle('get-log-history',  () => [...logBuffer])
ipcMain.handle('get-version',     () => require(path.join(__dirname, '..', 'package.json')).version)

ipcMain.handle('reset-app-data', async () => {
  if (!app.isPackaged) { console.warn('reset-app-data blocked in dev mode'); return }
  try { await fetch(`http://127.0.0.1:${BACKEND_PORT}/shutdown`, { method: 'POST' }) } catch {}
  await new Promise(r => setTimeout(r, 1000))
  try { process.kill(-backendProc.pid, 'SIGTERM') } catch {}
  fs.rmSync(USER_DATA, { recursive: true, force: true })
  app.quit()
})

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

// ── bootstrap IPC ─────────────────────────────────────────────────────────────
ipcMain.handle('needs-bootstrap', () => !fs.existsSync(VENV_PYTHON))

ipcMain.handle('run-bootstrap', () => new Promise((resolve, reject) => {
  const proc = spawn('bash', [path.join(APP_ROOT, 'scripts', 'bootstrap.sh'), path.join(USER_DATA, 'venv')], {
    cwd: APP_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, ONICHAT_UV: path.join(APP_ROOT, 'scripts', 'bin', 'uv') },
  })
  const send = (d) => String(d).split('\n').forEach(line => {
    if (line.trim() && mainWindow && !mainWindow.isDestroyed())
      mainWindow.webContents.send('bootstrap-log', line)
  })
  proc.stdout.on('data', send)
  proc.stderr.on('data', send)
  proc.on('close', (code) => code === 0 ? resolve() : reject(new Error(`Bootstrap failed (${code})`)))
}))

// ── kill any stale process holding the backend port ───────────────────────────
function killPortOwner(port) {
  return new Promise((resolve) => {
    const { execSync } = require('child_process')

    // Kill all PIDs holding the port (SIGKILL — no grace period needed here)
    try {
      const out = execSync(`lsof -ti :${port}`, { encoding: 'utf8' }).trim()
      const pids = out.split('\n').filter(Boolean).map(Number)
      pids.forEach((pid) => { try { process.kill(pid, 'SIGKILL') } catch {} })
    } catch {
      return resolve() // lsof found nothing
    }

    // Poll until the port is actually free (up to 3 s)
    const deadline = Date.now() + 3000
    const poll = () => {
      try {
        execSync(`lsof -ti :${port}`, { encoding: 'utf8', stdio: 'pipe' }).trim()
        // still occupied
        if (Date.now() < deadline) setTimeout(poll, 200)
        else resolve()
      } catch {
        resolve() // lsof exited non-zero = no owner = port is free
      }
    }
    setTimeout(poll, 200)
  })
}

// ── spawn backend ─────────────────────────────────────────────────────────────
function startBackend() {
  backendProc = spawn(VENV_PYTHON, ['-u', '-m', 'uvicorn', 'backend.main:app',
    '--port', String(BACKEND_PORT), '--host', '127.0.0.1'], {
    cwd: APP_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,   // new process group (PGID = PID) — lets us kill the whole tree
    env: {
      ...process.env,
      ONICHAT_APP_ROOT:  APP_ROOT,
      ONICHAT_USER_DATA: USER_DATA,
    },
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
  mainWindow.loadFile(path.join(APP_ROOT, 'app', 'loader.html'))
  mainWindow.on('closed', () => { mainWindow = null })
}

// ── start-backend IPC (called by loader.html after bootstrap) ─────────────────
ipcMain.handle('start-backend', async () => {
  if (!backendProc) {
    await Promise.all([killPortOwner(BACKEND_PORT), killPortOwner(50021)])
    startBackend()
  }
  await new Promise((resolve, reject) => pollHealth(resolve, reject))
})

// ── app lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(() => { createWindow() })

app.on('window-all-closed', () => app.quit())

app.on('will-quit', () => {
  // Synchronous fallback — fires even when before-quit is skipped.
  // before-quit already handles the graceful path; this just ensures the
  // process group is gone if we get here without it having run.
  if (!backendProc) return
  try { process.kill(-backendProc.pid, 'SIGTERM') } catch {}
})

app.on('before-quit', (e) => {
  if (!backendProc) return
  e.preventDefault()
  // 1. Ask the backend to flush cleanly (unload models, close sockets).
  fetch(`http://127.0.0.1:${BACKEND_PORT}/shutdown`, { method: 'POST' }).catch(() => {})
  // 2. Give it a moment, then kill the entire process group — uvicorn, llama-server,
  //    and voicevox all share the same PGID so one signal takes down the whole tree.
  setTimeout(() => {
    try { process.kill(-backendProc.pid, 'SIGTERM') } catch {}
    backendProc = null
    app.quit()
  }, 1000)
})
