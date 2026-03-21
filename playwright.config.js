const { defineConfig } = require('@playwright/test')

module.exports = defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  retries: 0,
  use: {
    headless: true,
    baseURL: 'http://127.0.0.1:8743',
  },
  // E2E requires the dev server to be running — start it manually with `npm start`
  // before running playwright tests.
  reporter: [['list'], ['html', { open: 'never' }]],
})
