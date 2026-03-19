"""
Manage a llama-server subprocess for local LLM inference.
Binary:  modules/llm/bin/llama-server
Models:  modules/llm/models/{key}.gguf
Port:    8745 (loopback only)
"""
import hashlib, json, os, socket, subprocess, time, urllib.request
from pathlib import Path
from typing import Callable, Optional

BIN_DIR    = Path(__file__).parent / "bin"
MODELS_DIR = Path(__file__).parent / "models"
LLAMA_BIN  = BIN_DIR / "llama-server"
LLAMA_PORT = 8745
LLAMA_URL  = f"http://127.0.0.1:{LLAMA_PORT}"

ProgressCb = Callable[[int, str], None]

# key → (local_filename, hf_repo, hf_filename, size_gb)
_CATALOG: dict[str, tuple[str, str, str, float]] = {
    "qwen3.5-2b-q4km":  ("qwen3.5-2b-q4_k_m.gguf",      "bartowski/Qwen_Qwen3.5-2B-GGUF",      "Qwen_Qwen3.5-2B-Q4_K_M.gguf",      1.33),
    "qwen3.5-4b-q4km":  ("qwen3.5-4b-q4_k_m.gguf",      "bartowski/Qwen_Qwen3.5-4B-GGUF",      "Qwen_Qwen3.5-4B-Q4_K_M.gguf",      2.87),
    "qwen3.5-9b-q4km":  ("qwen3.5-9b-q4_k_m.gguf",      "bartowski/Qwen_Qwen3.5-9B-GGUF",      "Qwen_Qwen3.5-9B-Q4_K_M.gguf",      5.89),
    "qwen3.5-27b-q4km": ("qwen3.5-27b-q4_k_m.gguf",     "bartowski/Qwen_Qwen3.5-27B-GGUF",     "Qwen_Qwen3.5-27B-Q4_K_M.gguf",     17.13),
    "qwen3.5-35b-q4km": ("qwen3.5-35b-a3b-q4_k_m.gguf", "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF", "Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf", 21.40),
}

# Exposed for backend to build _LOCAL_LLM_KEYS
MODEL_FILES: dict[str, str] = {k: v[0] for k, v in _CATALOG.items()}


class LlmManager:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._active_key: Optional[str] = None

    # ── queries ───────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def model_path(self, key: str) -> Optional[Path]:
        fname = MODEL_FILES.get(key)
        return (MODELS_DIR / fname) if fname else None

    def is_model_present(self, key: str) -> bool:
        p = self.model_path(key)
        return p is not None and p.exists()

    def models_status(self) -> list[dict]:
        return [
            {
                "key":     k,
                "label":   v[2].replace(".gguf", "").replace("-", " "),
                "size_gb": v[3],
                "present": (MODELS_DIR / v[0]).exists(),
                "running": self._active_key == k and self.is_running(),
            }
            for k, v in _CATALOG.items()
        ]

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, key: str) -> None:
        """Start llama-server for the given model key. Blocks until healthy (≤60 s)."""
        if self.is_running() and self._active_key == key:
            return
        self.stop()

        path = self.model_path(key)
        if not path or not path.exists():
            raise RuntimeError(f"Model file not found: {path}")
        if not LLAMA_BIN.exists():
            raise RuntimeError("llama-server binary not found — run the installer first")

        self._proc = subprocess.Popen(
            [
                str(LLAMA_BIN),
                "--model", str(path),
                "--host", "127.0.0.1",
                "--port", str(LLAMA_PORT),
                "--ctx-size", "4096",
                "--n-gpu-layers", "-1",   # offload all to Metal
                "--log-disable",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._active_key = key

        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("llama-server exited unexpectedly during startup")
            try:
                with urllib.request.urlopen(f"{LLAMA_URL}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)

        self.stop()
        raise RuntimeError("llama-server did not become healthy within 60 s")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self._active_key = None

    # ── model download ────────────────────────────────────────────────────────

    def download_model(self, key: str, progress: Optional[ProgressCb] = None) -> None:
        """Download a model GGUF from Hugging Face. Blocks — run via asyncio.to_thread()."""
        entry = _CATALOG.get(key)
        if entry is None:
            raise ValueError(f"Unknown model key: {key!r}")

        fname, hf_repo, hf_file, _ = entry
        url  = f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}"
        dest = MODELS_DIR / fname
        tmp  = dest.with_suffix(".part")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        def report(pct: int, msg: str):
            if progress:
                progress(pct, msg)

        # ── Fetch expected SHA-256 from HF tree API ────────────────────────────
        report(0, "Fetching file metadata…")
        expected_sha: Optional[str] = None
        try:
            api_url = f"https://huggingface.co/api/models/{hf_repo}/tree/main"
            api_req = urllib.request.Request(api_url, headers={"User-Agent": "OniChat/1.0"})
            with urllib.request.urlopen(api_req, timeout=10) as r:
                for item in json.loads(r.read()):
                    if item.get("path") == hf_file:
                        expected_sha = item.get("lfs", {}).get("sha256")
                        break
        except Exception:
            pass   # proceed without hash check if API unreachable

        # ── Download ───────────────────────────────────────────────────────────
        _retryable = (socket.gaierror, ConnectionResetError, TimeoutError, urllib.error.URLError)
        def _urlopen_retry(u, timeout, retries=3):
            req = urllib.request.Request(u, headers={"User-Agent": "OniChat/1.0"})
            last_err = None
            for attempt in range(retries):
                try:
                    return urllib.request.urlopen(req, timeout=timeout)
                except _retryable as e:
                    last_err = e
                    if attempt < retries - 1:
                        report(0, f"Network error, retrying ({attempt+2}/{retries})…")
                        time.sleep(2 * (attempt + 1))
            raise last_err

        report(0, "Connecting to Hugging Face…")
        try:
            with _urlopen_retry(url, timeout=600) as r:
                size    = int(r.headers.get("Content-Length", 0))
                size_gb = size / 1e9
                dl      = 0
                hasher  = hashlib.sha256()
                with open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)   # 1 MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        hasher.update(chunk)
                        dl += len(chunk)
                        if size:
                            report(int(dl / size * 99), f"Downloading… {dl/1e9:.2f}/{size_gb:.1f} GB")
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        # ── Verify SHA-256 ─────────────────────────────────────────────────────
        report(99, "Verifying integrity…")
        actual_sha = hasher.hexdigest()
        if expected_sha and actual_sha != expected_sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch — file may be corrupt or tampered.\n"
                f"  expected: {expected_sha}\n"
                f"  got:      {actual_sha}"
            )

        tmp.rename(dest)
        report(100, f"Ready: {dest.name}")


llm_manager = LlmManager()
