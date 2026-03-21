"""
Patch heavy module-level imports before backend.main is loaded.
Must run before any test that imports backend.main.
"""
import sys
import numpy as np
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── onnxruntime mock ──────────────────────────────────────────────────────────
# Prevents ONNX InferenceSession from requiring a real model file.
mock_ort = MagicMock()
_mock_vad_session = MagicMock()
_mock_vad_session.get_inputs.return_value = [
    MagicMock(name="input"),
    MagicMock(name="sr"),
    MagicMock(name="state"),
]
_mock_vad_session.run.return_value = [
    np.zeros((1, 1, 1), dtype=np.float32),   # speech prob ≈ 0 (silence)
    np.zeros((2, 1, 128), dtype=np.float32),  # new hidden state
]
mock_ort.InferenceSession.return_value = _mock_vad_session
sys.modules["onnxruntime"] = mock_ort

# ── prevent VAD model download ────────────────────────────────────────────────
import urllib.request as _ur
_orig_urlretrieve = _ur.urlretrieve
def _no_download(url, dst, *a, **kw):
    # Create a dummy file so the existence check passes
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).touch()
_ur.urlretrieve = _no_download

# ── soundfile / scipy stubs (only needed if not installed in test venv) ───────
# Uncomment if test venv lacks these packages:
# sys.modules.setdefault("soundfile", MagicMock())
# sys.modules.setdefault("scipy", MagicMock())
# sys.modules.setdefault("scipy.signal", MagicMock())

# ── modules/* stubs ───────────────────────────────────────────────────────────
# Replace heavy module dependencies with lightweight fakes.

_fake_stt_manager = MagicMock()
_fake_stt_manager.loaded_model = None
_fake_stt_manager.active_key = "qwen3-1.7b-4bit"
_fake_stt_manager.transcribe = MagicMock(return_value="こんにちは")

_fake_tts_manager = MagicMock()
_fake_tts_manager.voicevox_tts = AsyncMock(return_value=b"RIFF\x00\x00\x00\x00WAVEfmt ")

_fake_llm_manager = MagicMock()
_fake_llm_manager.is_running.return_value = True
_fake_llm_manager._active_key = "qwen3-4b-4bit"
_fake_llm_manager.is_model_present.return_value = True

_fake_voicevox_manager = MagicMock()
_fake_voicevox_manager.is_running.return_value = True
_fake_voicevox_manager.is_installed.return_value = True
_fake_voicevox_manager.endpoint = "http://127.0.0.1:50021"

# Patch module imports
sys.modules.setdefault("modules.tts.tts_manager", MagicMock(
    voicevox_tts=AsyncMock(return_value=b"RIFF\x00\x00\x00\x00WAVEfmt ")
))
sys.modules.setdefault("modules.stt.stt_manager", MagicMock(
    stt_manager=_fake_stt_manager,
    MODELS={"qwen3-1.7b-4bit": {"label": "Qwen3 1.7B", "backend": "mlx", "repo": None}},
    AVAILABLE={"mlx": True},
    DEFAULT_MODEL="qwen3-1.7b-4bit",
))
sys.modules.setdefault("modules.llm.llm_manager", MagicMock(
    llm_manager=_fake_llm_manager,
    MODEL_FILES={"qwen3-4b-4bit": "qwen3-4b-4bit.gguf"},
    LLAMA_URL="http://127.0.0.1:8080",
))
sys.modules.setdefault("modules.tts.voicevox_manager", MagicMock(
    voicevox_manager=_fake_voicevox_manager,
))
sys.modules.setdefault("modules.system.system_info", MagicMock(
    get_system_info=AsyncMock(return_value={}),
))
sys.modules.setdefault("modules.llm.installer", MagicMock(
    is_installed=MagicMock(return_value=True),
    install=MagicMock(),
))
sys.modules.setdefault("modules.tts.installer", MagicMock(
    is_installed=MagicMock(return_value=True),
    install=MagicMock(),
))
