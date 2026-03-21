/**
 * Tests for the model selection / doDownloads bug.
 *
 * Bug (fixed): After repopulatePresence() was called, restoreSelections(_settings)
 * was called with an empty llm_model, triggering smartLlmDefault to pick 9B instead
 * of the user's 4B choice.
 *
 * Fix: Preserve dropdown values BEFORE repopulatePresence() and restore them AFTER,
 * without calling restoreSelections.
 *
 * Run: npx jest tests/frontend/model_selection.test.js
 */

// ── minimal DOM ──────────────────────────────────────────────────────────────

document.body.innerHTML = `
  <select id="llm-model-select">
    <option value="">— select —</option>
    <option value="qwen3-4b-4bit">Qwen3 4B</option>
    <option value="qwen3-9b-4bit">Qwen3 9B</option>
  </select>
  <select id="stt-model-select">
    <option value="qwen3-1.7b-4bit">Qwen3 1.7B</option>
    <option value="remote">Remote</option>
  </select>
`;

function getLlmSelect() { return document.getElementById('llm-model-select'); }
function getSttSelect() { return document.getElementById('stt-model-select'); }

// ── simulate the doDownloads pattern ─────────────────────────────────────────

/**
 * BUGGY version: calls restoreSelections after repopulate, which resets to default
 */
function doDownloadsBuggy(modelsFromServer, settings) {
  const llmSel = getLlmSelect();
  const sttSel = getSttSelect();

  // Simulate repopulatePresence: re-renders options
  llmSel.innerHTML = '';
  modelsFromServer.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = m.label;
    llmSel.appendChild(opt);
  });

  // BUG: restoreSelections uses settings.llm_model which may be empty,
  // causing smartLlmDefault to pick the largest available model
  restoreSelectionsBuggy(settings);
}

function restoreSelectionsBuggy(settings) {
  const llmSel = getLlmSelect();
  const key = settings.llm_model || smartLlmDefault(llmSel);
  if ([...llmSel.options].some(o => o.value === key)) {
    llmSel.value = key;
  }
}

function smartLlmDefault(select) {
  // Picks the largest downloaded model — simulates the 9B preference
  const opts = [...select.options].filter(o => o.value.includes('9b'));
  return opts[0]?.value || select.options[0]?.value || '';
}

/**
 * FIXED version: snapshot values before repopulate, restore after without
 * going through restoreSelections.
 */
function doDownloadsFixed(modelsFromServer, settings) {
  const llmSel = getLlmSelect();
  const sttSel = getSttSelect();

  // ✅ FIX: snapshot current selections before repopulate
  const prevLlm = llmSel.value;
  const prevStt = sttSel.value;

  // Simulate repopulatePresence
  llmSel.innerHTML = '';
  modelsFromServer.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = m.label;
    llmSel.appendChild(opt);
  });

  // ✅ FIX: restore snapshot values, not settings (which may have empty llm_model)
  if (prevLlm && [...llmSel.options].some(o => o.value === prevLlm)) {
    llmSel.value = prevLlm;
  }
  if (prevStt && [...sttSel.options].some(o => o.value === prevStt)) {
    sttSel.value = prevStt;
  }
}

// ── fixtures ─────────────────────────────────────────────────────────────────

const MODELS_FROM_SERVER = [
  { key: 'qwen3-4b-4bit', label: 'Qwen3 4B', present: true },
  { key: 'qwen3-9b-4bit', label: 'Qwen3 9B', present: true },
];

// Settings with empty llm_model (fresh install or not yet saved)
const SETTINGS_EMPTY_MODEL = { llm_model: '', stt_model: 'qwen3-1.7b-4bit' };

beforeEach(() => {
  // User has selected 4B
  getLlmSelect().value = 'qwen3-4b-4bit';
  getSttSelect().value = 'qwen3-1.7b-4bit';
});

// ── tests ─────────────────────────────────────────────────────────────────────

describe('doDownloads model selection bug', () => {
  test('BUGGY: restoreSelections resets to 9B when llm_model is empty', () => {
    getLlmSelect().value = 'qwen3-4b-4bit';
    doDownloadsBuggy(MODELS_FROM_SERVER, SETTINGS_EMPTY_MODEL);
    // This demonstrates the bug: user picked 4B but gets 9B
    expect(getLlmSelect().value).toBe('qwen3-9b-4bit');
  });

  test('FIXED: snapshot preserves user-selected 4B across repopulate', () => {
    getLlmSelect().value = 'qwen3-4b-4bit';
    doDownloadsFixed(MODELS_FROM_SERVER, SETTINGS_EMPTY_MODEL);
    expect(getLlmSelect().value).toBe('qwen3-4b-4bit');
  });

  test('FIXED: snapshot preserves 9B if user explicitly chose it', () => {
    getLlmSelect().value = 'qwen3-9b-4bit';
    doDownloadsFixed(MODELS_FROM_SERVER, SETTINGS_EMPTY_MODEL);
    expect(getLlmSelect().value).toBe('qwen3-9b-4bit');
  });

  test('FIXED: falls back gracefully if snapshotted value disappears from list', () => {
    getLlmSelect().value = 'qwen3-4b-4bit';
    // New server response doesn't include 4B (e.g. it was uninstalled)
    const modelsWithout4b = [{ key: 'qwen3-9b-4bit', label: 'Qwen3 9B', present: true }];
    expect(() => doDownloadsFixed(modelsWithout4b, SETTINGS_EMPTY_MODEL)).not.toThrow();
    // Whatever value ends up selected, it should be a valid option
    const validValues = [...getLlmSelect().options].map(o => o.value);
    expect(validValues).toContain(getLlmSelect().value);
  });
});

// ── settings round-trip through _hiddenSettings ───────────────────────────────

describe('_hiddenSettings round-trip', () => {
  /**
   * Bug: when saveEndpoints() was called, non-UI settings fields (like llm_api_key,
   * voice_en, etc.) were lost because they weren't in the DOM.
   *
   * Fix: _hiddenSettings stores the full last-read blob; saveEndpoints spreads it
   * before adding UI-visible fields.
   */

  let _hiddenSettings = {};

  function loadSettings(fullBlob) {
    _hiddenSettings = fullBlob;
    // Only set UI-visible fields
    document.getElementById('llm-model-select').value = fullBlob.llm_model || '';
  }

  function saveSettings(uiValues) {
    return { ..._hiddenSettings, ...uiValues };
  }

  test('non-UI fields survive a load→save round-trip', () => {
    const original = {
      llm_model: 'qwen3-4b-4bit',
      llm_api_key: 'sk-secret',
      voice_en: 'Ryan',
      voice_ja: 'Ono_Anna',
      system_prompt_en: 'Be concise.',
      // ... many more fields not shown in UI
    };

    loadSettings(original);

    const saved = saveSettings({
      llm_model: getLlmSelect().value,
      language: 'ja',
    });

    expect(saved.llm_api_key).toBe('sk-secret');
    expect(saved.voice_en).toBe('Ryan');
    expect(saved.system_prompt_en).toBe('Be concise.');
  });

  test('UI values override hidden settings on save', () => {
    loadSettings({ language: 'en', llm_model: 'qwen3-9b-4bit' });
    const saved = saveSettings({ language: 'zh' });
    expect(saved.language).toBe('zh');
  });
});
