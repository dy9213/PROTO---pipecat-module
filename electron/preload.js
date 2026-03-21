const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('api', {
  getSettings:    ()   => ipcRenderer.invoke('get-settings'),
  saveSettings:   (s)  => ipcRenderer.invoke('save-settings', s),
  backendPort:    ipcRenderer.sendSync('get-backend-port-sync'),
  needsBootstrap:  ()   => ipcRenderer.invoke('needs-bootstrap'),
  runBootstrap:    ()   => ipcRenderer.invoke('run-bootstrap'),
  onBootstrapLog:  (cb) => ipcRenderer.on('bootstrap-log', (_, line) => cb(line)),
  startBackend:    ()   => ipcRenderer.invoke('start-backend'),
  getLogHistory:   ()   => ipcRenderer.invoke('get-log-history'),
  onLog:          (cb) => ipcRenderer.on('log', (_, entry) => cb(entry)),
  getVersion:     ()   => ipcRenderer.invoke('get-version'),
  resetAppData:   ()   => ipcRenderer.invoke('reset-app-data'),
})
