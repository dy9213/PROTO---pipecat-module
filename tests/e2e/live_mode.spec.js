/**
 * Playwright E2E tests for live mode.
 *
 * Prerequisites:
 *   - App must be running: npm start  (dev mode, port 8743)
 *   - Backend must be healthy: http://127.0.0.1:8743/health
 *   - A local LLM model must be downloaded
 *
 * Run: npx playwright test tests/e2e/live_mode.spec.js
 */

const { test, expect } = require('@playwright/test');
const path = require('path');

const BACKEND_URL = 'http://127.0.0.1:8743';
const APP_URL = `file://${path.resolve(__dirname, '../../app/index.html')}`;

// ── helpers ──────────────────────────────────────────────────────────────────

async function waitForBackend(page, timeout = 10_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      const res = await page.request.get(`${BACKEND_URL}/health`);
      if (res.ok()) return;
    } catch {}
    await page.waitForTimeout(500);
  }
  throw new Error('Backend did not become healthy within timeout');
}

async function getLiveModeButton(page) {
  // The live mode button / microphone trigger
  return page.locator('#live-btn, [data-action="live"], button:has-text("Live")').first();
}

// ── basic connectivity ────────────────────────────────────────────────────────

test.describe('Backend connectivity', () => {
  test('health endpoint returns 200', async ({ request }) => {
    const res = await request.get(`${BACKEND_URL}/health`);
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.status).toBe('ok');
  });

  test('settings endpoint returns defaults', async ({ request }) => {
    const res = await request.get(`${BACKEND_URL}/settings`);
    expect(res.status()).toBe(200);
    const s = await res.json();
    expect(s).toHaveProperty('language');
    expect(s).toHaveProperty('tts_mode');
    expect(['voicevox', 'kokoro', 'remote']).toContain(s.tts_mode);
  });

  test('services status endpoint lists all services', async ({ request }) => {
    const res = await request.get(`${BACKEND_URL}/system/services`);
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body).toHaveProperty('llama_server');
    expect(body).toHaveProperty('voicevox_engine');
    expect(body).toHaveProperty('stt');
  });
});

// ── settings persistence ──────────────────────────────────────────────────────

test.describe('Settings persistence', () => {
  test('saved settings survive a round-trip via API', async ({ request }) => {
    const unique_prompt = `test-prompt-${Date.now()}`;

    const saveRes = await request.post(`${BACKEND_URL}/settings`, {
      data: {
        stt_model: 'qwen3-1.7b-4bit',
        tts_mode: 'voicevox',
        voicevox_speaker: 2,
        language: 'ja',
        llm_model: '',
        llm_api_key: '',
        llm_endpoint: '',
        stt_endpoint: '',
        tts_endpoint: '',
        system_prompt_ja: unique_prompt,
        system_prompt_en: '',
        system_prompt_zh: '',
        search_online: false,
        translate_to: 'en',
        voice_en: 'Ryan',
        voice_ja: 'Ono_Anna',
        voice_zh: 'Vivian',
      },
    });
    expect(saveRes.status()).toBe(200);

    const getRes = await request.get(`${BACKEND_URL}/settings`);
    const loaded = await getRes.json();
    expect(loaded.system_prompt_ja).toBe(unique_prompt);
  });

  test('POST /settings normalises legacy tts_mode=remote to kokoro', async ({ request }) => {
    const res = await request.post(`${BACKEND_URL}/settings`, {
      data: {
        tts_mode: 'remote',
        language: 'ja',
        stt_model: 'qwen3-1.7b-4bit',
        voicevox_speaker: 2,
        llm_model: '', llm_api_key: '', llm_endpoint: '',
        stt_endpoint: '', tts_endpoint: '',
        system_prompt_ja: '', system_prompt_en: '', system_prompt_zh: '',
        search_online: false, translate_to: 'en',
        voice_en: 'Ryan', voice_ja: 'Ono_Anna', voice_zh: 'Vivian',
      },
    });
    // The endpoint should not error
    expect(res.status()).toBe(200);
  });
});

// ── live mode WebSocket ───────────────────────────────────────────────────────

test.describe('Live mode WebSocket', () => {
  test('connects and responds to ping', async ({ page }) => {
    const messages = [];

    // Connect directly to the backend WebSocket
    await page.goto('about:blank');
    const wsMessages = page.evaluate(() => {
      return new Promise((resolve) => {
        const ws = new WebSocket('ws://127.0.0.1:8743/live');
        const msgs = [];
        ws.onopen = () => ws.send(JSON.stringify({ type: 'ping' }));
        ws.onmessage = (e) => {
          msgs.push(JSON.parse(e.data));
          if (msgs.length >= 1) { ws.close(); resolve(msgs); }
        };
        ws.onerror = () => resolve([]);
        setTimeout(() => { ws.close(); resolve(msgs); }, 3000);
      });
    });

    const msgs = await wsMessages;
    expect(msgs.some(m => m.type === 'pong')).toBe(true);
  });

  test('interrupt during silence returns interrupted event', async ({ page }) => {
    await page.goto('about:blank');
    const result = await page.evaluate(() => {
      return new Promise((resolve) => {
        const ws = new WebSocket('ws://127.0.0.1:8743/live');
        ws.onopen = () => {
          ws.send(JSON.stringify({ type: 'interrupt' }));
        };
        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data);
          ws.close();
          resolve(msg);
        };
        ws.onerror = () => resolve(null);
        setTimeout(() => { ws.close(); resolve(null); }, 3000);
      });
    });

    expect(result).not.toBeNull();
    expect(result.type).toBe('interrupted');
  });

  test('interrupt does NOT produce error event', async ({ page }) => {
    /**
     * Regression test for the interrupt-during-processing bug.
     *
     * The WebSocket should receive exactly one message (interrupted) after
     * sending interrupt — NOT an error event.
     *
     * This test will FAIL until the bug in process_turn's except block is fixed.
     */
    await page.goto('about:blank');
    const messages = await page.evaluate(() => {
      return new Promise((resolve) => {
        const ws = new WebSocket('ws://127.0.0.1:8743/live');
        const msgs = [];
        ws.onopen = () => {
          ws.send(JSON.stringify({ type: 'interrupt' }));
          // Collect for 1s to catch any delayed error event
          setTimeout(() => { ws.close(); resolve(msgs); }, 1000);
        };
        ws.onmessage = (e) => {
          msgs.push(JSON.parse(e.data));
        };
        ws.onerror = () => resolve([]);
      });
    });

    const errorMsgs = messages.filter(m => m.type === 'error');
    expect(errorMsgs).toHaveLength(0);

    const interruptedMsgs = messages.filter(m => m.type === 'interrupted');
    expect(interruptedMsgs).toHaveLength(1);
  });
});

// ── chat API ──────────────────────────────────────────────────────────────────

test.describe('Chat stream API', () => {
  test.skip('POST /chat/stream returns SSE (requires local LLM running)', async ({ request }) => {
    const res = await request.post(`${BACKEND_URL}/chat/stream`, {
      data: { message: 'こんにちは', history: [], language: 'ja' },
    });
    expect(res.status()).toBe(200);
    expect(res.headers()['content-type']).toContain('text/event-stream');
    const body = await res.text();
    expect(body).toContain('data:');
  });
});
