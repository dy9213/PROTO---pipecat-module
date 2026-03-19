const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('api', {
  getSettings:    ()   => ipcRenderer.invoke('get-settings'),
  saveSettings:   (s)  => ipcRenderer.invoke('save-settings', s),
  getLogHistory:  ()   => ipcRenderer.invoke('get-log-history'),
  onLog:          (cb) => ipcRenderer.on('log', (_, entry) => cb(entry)),
})
