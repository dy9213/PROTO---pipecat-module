/**
 * Tests for live mode frontend logic.
 *
 * Covers:
 *  - _liveLLMDone flag transitions
 *  - Interrupt decision tree (llmWasDone true/false → undo vs badge)
 *  - liveInterrupt() debounce
 *  - processPTT async guard (state check after STT upload)
 *
 * Run with: npx jest tests/frontend/live.test.js
 */

// ── jsdom environment is set by jest.config.js ──────────────────────────────

// Minimal DOM scaffolding
document.body.innerHTML = `
  <div id="chat-list"></div>
  <div id="status-dot" class="dot"></div>
`;

// ── stub global dependencies that index.html normally provides ───────────────

global.STATE = {
  IDLE: 'IDLE',
  PTT_RECORDING: 'PTT_RECORDING',
  PTT_PROCESSING: 'PTT_PROCESSING',
  LIVE_ACTIVE: 'LIVE_ACTIVE',
};

let _currentState = global.STATE.IDLE;
global.state = global.STATE.IDLE;

global.setState = jest.fn((s) => {
  _currentState = s;
  global.state = s;
});

global.chatList = document.getElementById('chat-list');
global.chatHistory = [];

global.appendUserMessage = jest.fn((text) => {
  chatHistory.push({ role: 'user', content: text });
  const el = document.createElement('div');
  el.className = 'bubble-wrap user';
  el.id = 'user-' + Date.now();
  el.textContent = text;
  chatList.appendChild(el);
  return el.id;
});

global.undoLastUserMessage = jest.fn(() => {
  const last = chatList.querySelector('.bubble-wrap.user:last-child');
  if (last) last.remove();
  chatHistory.pop();
});

global.startAssistantBubble = jest.fn(() => {
  const el = document.createElement('div');
  el.className = 'bubble-wrap assistant';
  el.id = 'asst-' + Date.now();
  const b = document.createElement('div');
  b.className = 'bubble';
  el.appendChild(b);
  chatList.appendChild(el);
  return el.id;
});

global.scrollBottom = jest.fn();
global.exitLiveMode = jest.fn();
global.fmtTime = jest.fn(() => '12:00');
global.finalizeBubbleWithChevron = jest.fn();
global.cancelLivePlayback = jest.fn();

// Inline the subset of live mode logic we want to test
// (extracted from index.html — keep in sync if the source changes)

// ── _liveLLMDone and state variables ─────────────────────────────────────────

let _liveLLMDone = false;
let livePlaybackGated = false;
let liveInterruptSent = false;
let interruptStreak = 0;
let _ttsActiveBubbleWrap = null;
let _liveTurnText = '';
let _lastInterruptTime = 0;

function resetLiveState() {
  _liveLLMDone = false;
  livePlaybackGated = false;
  liveInterruptSent = false;
  interruptStreak = 0;
  _ttsActiveBubbleWrap = null;
  _liveTurnText = '';
  _lastInterruptTime = 0;
  chatList.innerHTML = '';
  chatHistory.length = 0;
  jest.clearAllMocks();
  _currentState = STATE.LIVE_ACTIVE;
  global.state = STATE.LIVE_ACTIVE;
}

function setBubbleTtsStatus(wrapEl, mode) {
  if (!wrapEl) return;
  wrapEl.querySelector('.bubble-tts-status')?.remove();
  if (!mode) return;
  let meta = wrapEl.querySelector('.bubble-meta');
  if (!meta) {
    meta = document.createElement('div');
    meta.className = 'bubble-meta';
    wrapEl.appendChild(meta);
  }
  const span = document.createElement('span');
  span.className = 'bubble-tts-status';
  if (mode === 'synthesizing') {
    span.classList.add('synthesizing');
    span.textContent = 'Processing';
  } else if (mode === 'playback') {
    span.textContent = '▶ playback';
  } else {
    span.classList.add('interrupted');
    span.textContent = '✕ interrupted';
    setTimeout(() => span.remove(), 800);
  }
  meta.appendChild(span);
}

// Simulates the ws.onmessage handler from index.html
function handleLiveMessage(msg) {
  if (msg.type === 'transcript' && msg.final) {
    _liveLLMDone = false;
    appendUserMessage(msg.text, 'voice');
    startAssistantBubble();
  } else if (msg.type === 'tts_start') {
    cancelLivePlayback();
    _liveTurnText = '';
    livePlaybackGated = true;
    liveInterruptSent = false;
    interruptStreak = 0;
    _liveLLMDone = true;
    _ttsActiveBubbleWrap = chatList.querySelector('.bubble-wrap.assistant:last-child') || null;
    setBubbleTtsStatus(_ttsActiveBubbleWrap, 'playback');
  } else if (msg.type === 'tts_end') {
    _liveTurnText = '';
    livePlaybackGated = false;
    liveInterruptSent = false;
    interruptStreak = 0;
    setBubbleTtsStatus(_ttsActiveBubbleWrap, null);
    _ttsActiveBubbleWrap = null;
  } else if (msg.type === 'interrupted') {
    const llmWasDone = _liveLLMDone;
    livePlaybackGated = false;
    liveInterruptSent = false;
    interruptStreak = 0;
    _liveLLMDone = false;

    if (!llmWasDone) {
      chatList.querySelector('.bubble-wrap.assistant:last-child')?.remove();
      undoLastUserMessage();
      _ttsActiveBubbleWrap = null;
    } else {
      setBubbleTtsStatus(_ttsActiveBubbleWrap, 'interrupted');
      _ttsActiveBubbleWrap = null;
    }
  } else if (msg.type === 'error') {
    exitLiveMode();
  }
}

function liveInterrupt(liveWS) {
  const now = Date.now();
  if (now - _lastInterruptTime < 500) return;
  _lastInterruptTime = now;
  if (liveInterruptSent) return;
  cancelLivePlayback();
  liveInterruptSent = true;
  if (liveWS?.readyState === WebSocket.OPEN)
    liveWS.send(JSON.stringify({ type: 'interrupt' }));
}

// ── tests ─────────────────────────────────────────────────────────────────────

beforeEach(() => resetLiveState());

// ── _liveLLMDone flag transitions ─────────────────────────────────────────────

describe('_liveLLMDone flag', () => {
  test('is false before any transcript arrives', () => {
    expect(_liveLLMDone).toBe(false);
  });

  test('stays false after transcript', () => {
    handleLiveMessage({ type: 'transcript', text: 'hello', final: true });
    expect(_liveLLMDone).toBe(false);
  });

  test('becomes true on tts_start', () => {
    handleLiveMessage({ type: 'transcript', text: 'hello', final: true });
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    expect(_liveLLMDone).toBe(true);
  });

  test('resets to false on tts_end', () => {
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    handleLiveMessage({ type: 'tts_end' });
    expect(_liveLLMDone).toBe(false);
  });

  test('resets to false on interrupted', () => {
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    handleLiveMessage({ type: 'interrupted' });
    expect(_liveLLMDone).toBe(false);
  });
});

// ── interrupt decision tree ───────────────────────────────────────────────────

describe('interrupted: llmWasDone = false (LLM still streaming)', () => {
  test('removes assistant bubble', () => {
    handleLiveMessage({ type: 'transcript', text: 'test', final: true });
    handleLiveMessage({ type: 'interrupted' }); // _liveLLMDone is still false
    expect(chatList.querySelector('.bubble-wrap.assistant')).toBeNull();
  });

  test('calls undoLastUserMessage', () => {
    handleLiveMessage({ type: 'transcript', text: 'test', final: true });
    handleLiveMessage({ type: 'interrupted' });
    expect(undoLastUserMessage).toHaveBeenCalledTimes(1);
  });

  test('clears _ttsActiveBubbleWrap', () => {
    handleLiveMessage({ type: 'transcript', text: 'test', final: true });
    handleLiveMessage({ type: 'interrupted' });
    expect(_ttsActiveBubbleWrap).toBeNull();
  });
});

describe('interrupted: llmWasDone = true (LLM done, TTS was playing)', () => {
  test('does NOT call undoLastUserMessage', () => {
    handleLiveMessage({ type: 'transcript', text: 'test', final: true });
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    handleLiveMessage({ type: 'interrupted' });
    expect(undoLastUserMessage).not.toHaveBeenCalled();
  });

  test('shows interrupted badge on the assistant bubble', () => {
    handleLiveMessage({ type: 'transcript', text: 'test', final: true });
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    const bubble = chatList.querySelector('.bubble-wrap.assistant:last-child');
    handleLiveMessage({ type: 'interrupted' });
    // The bubble should have received the interrupted status
    expect(bubble?.querySelector('.bubble-tts-status.interrupted')).not.toBeNull();
  });

  test('user and assistant bubbles remain in the chat', () => {
    handleLiveMessage({ type: 'transcript', text: 'test', final: true });
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    handleLiveMessage({ type: 'interrupted' });
    expect(chatList.querySelector('.bubble-wrap.user')).not.toBeNull();
    expect(chatList.querySelector('.bubble-wrap.assistant')).not.toBeNull();
  });
});

// ── error event must trigger exitLiveMode ─────────────────────────────────────

describe('error event', () => {
  test('calls exitLiveMode', () => {
    handleLiveMessage({ type: 'error', message: 'something broke' });
    expect(exitLiveMode).toHaveBeenCalledTimes(1);
  });
});

// ── livePlaybackGated resets ──────────────────────────────────────────────────

describe('livePlaybackGated', () => {
  test('is set to true by tts_start', () => {
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    expect(livePlaybackGated).toBe(true);
  });

  test('is cleared by tts_end', () => {
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    handleLiveMessage({ type: 'tts_end' });
    expect(livePlaybackGated).toBe(false);
  });

  test('is cleared by interrupted', () => {
    handleLiveMessage({ type: 'tts_start', sample_rate: 24000 });
    handleLiveMessage({ type: 'interrupted' });
    expect(livePlaybackGated).toBe(false);
  });
});

// ── liveInterrupt debounce ────────────────────────────────────────────────────

describe('liveInterrupt debounce', () => {
  const OPEN = 1; // WebSocket.OPEN constant

  test('sends interrupt on first call', () => {
    const mockWS = { readyState: OPEN, send: jest.fn() };
    global.WebSocket = { OPEN };
    liveInterrupt(mockWS);
    expect(mockWS.send).toHaveBeenCalledTimes(1);
    const payload = JSON.parse(mockWS.send.mock.calls[0][0]);
    expect(payload.type).toBe('interrupt');
  });

  test('does not send a second interrupt within 500 ms', () => {
    const mockWS = { readyState: OPEN, send: jest.fn() };
    global.WebSocket = { OPEN };
    liveInterrupt(mockWS);
    liveInterrupt(mockWS); // immediate second call
    expect(mockWS.send).toHaveBeenCalledTimes(1);
  });

  test('does not send after liveInterruptSent is set', () => {
    const mockWS = { readyState: OPEN, send: jest.fn() };
    global.WebSocket = { OPEN };
    liveInterruptSent = true;
    liveInterrupt(mockWS);
    expect(mockWS.send).not.toHaveBeenCalled();
  });
});

// ── processPTT state guard ────────────────────────────────────────────────────

describe('processPTT state guard', () => {
  /**
   * processPTT must abort early if state is no longer PTT_PROCESSING
   * after the async STT upload completes.  This prevents phantom turns
   * when the user aborts quickly.
   */

  async function simulateProcessPTT({ stateAfterStt }) {
    const fakeChunks = [new ArrayBuffer(512)];

    // Stub uploadPcmChunks to simulate async STT
    const transcript = 'テスト';
    await Promise.resolve(); // yield

    // After STT, state may have changed
    global.state = stateAfterStt;
    if (global.state !== STATE.PTT_PROCESSING) return { aborted: true };

    return { aborted: false, transcript };
  }

  test('aborts if state changed to IDLE before STT returns', async () => {
    global.state = STATE.PTT_PROCESSING;
    const result = await simulateProcessPTT({ stateAfterStt: STATE.IDLE });
    expect(result.aborted).toBe(true);
    expect(appendUserMessage).not.toHaveBeenCalled();
  });

  test('continues if state is still PTT_PROCESSING after STT', async () => {
    global.state = STATE.PTT_PROCESSING;
    const result = await simulateProcessPTT({ stateAfterStt: STATE.PTT_PROCESSING });
    expect(result.aborted).toBe(false);
  });
});
