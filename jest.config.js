/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'jsdom',
  testMatch: ['**/tests/frontend/**/*.test.js'],
  // Suppress jsdom navigation/resource warnings
  testEnvironmentOptions: {
    url: 'http://localhost',
  },
  // Show individual test names
  verbose: true,
}
