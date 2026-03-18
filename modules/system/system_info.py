"""
System resource scanner for the loader screen.
Requires: pip install psutil
"""
import subprocess

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ── Model catalog ─────────────────────────────────────────────────────────────
# size_gb: approximate unified memory footprint when loaded

STT_CATALOG = [
    {"key": "qwen3-0.6b-4bit", "label": "Qwen3-ASR 0.6B 4bit",   "size_gb": 0.3},
    {"key": "qwen3-0.6b-8bit", "label": "Qwen3-ASR 0.6B 8bit",   "size_gb": 0.4},
    {"key": "qwen3-0.6b-bf16", "label": "Qwen3-ASR 0.6B bf16",   "size_gb": 0.8},
    {"key": "qwen3-1.7b-4bit", "label": "Qwen3-ASR 1.7B 4bit",   "size_gb": 0.6},
    {"key": "qwen3-1.7b-8bit", "label": "Qwen3-ASR 1.7B 8bit",   "size_gb": 1.8},
    {"key": "qwen3-1.7b-bf16", "label": "Qwen3-ASR 1.7B bf16",   "size_gb": 3.4},
    {"key": "kotoba-whisper",  "label": "Kotoba Whisper v2",      "size_gb": 1.5},
    {"key": "remote",          "label": "Remote (OpenAI-compat)", "size_gb": 0.0},
]

TTS_CATALOG = [
    {"key": "mlx",      "label": "MLX Audio TTS",         "size_gb": 0.5},
    {"key": "voicevox", "label": "VOICEVOX (local core)",  "size_gb": 0.3},
    {"key": "remote",   "label": "Remote (OpenAI-compat)", "size_gb": 0.0},
]

LLM_CATALOG = [
    {"key": "remote",         "label": "Remote endpoint",   "size_gb": 0.0},
    {"key": "qwen3-4b-q4km",  "label": "Qwen3-4B Q4_K_M",  "size_gb": 2.6},
    {"key": "qwen3-8b-q4km",  "label": "Qwen3-8B Q4_K_M",  "size_gb": 5.0},
    {"key": "qwen3-14b-q4km", "label": "Qwen3-14B Q4_K_M", "size_gb": 9.0},
    {"key": "qwen3-32b-q4km", "label": "Qwen3-32B Q4_K_M", "size_gb": 20.0},
]

_IGNORE = {
    "kernel_task", "launchd", "WindowServer", "loginwindow",
    "kextd", "notifyd", "configd", "sysmond", "logd", "opendirectoryd",
    "diskarbitrationd", "cfprefsd", "coreauthd", "powerd", "airportd",
}


def _memory_pressure() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "vm.memory_pressure"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        val = int(out.split(":")[-1].strip())
        return {1: "normal", 2: "warn", 4: "critical"}.get(val, "unknown")
    except Exception:
        return "unknown"


def get_system_info(n_procs: int = 5) -> dict:
    """Return memory stats, top processes, and model catalog."""
    if not _PSUTIL:
        return {"error": "psutil not installed — run: pip install psutil"}

    vm = psutil.virtual_memory()
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            name = (p.info["name"] or "").strip()
            if not name or name in _IGNORE:
                continue
            rss_gb = p.info["memory_info"].rss / 1e9
            if rss_gb < 0.05:
                continue
            procs.append({"pid": p.info["pid"], "name": name, "mem_gb": round(rss_gb, 2)})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    procs.sort(key=lambda x: x["mem_gb"], reverse=True)

    return {
        "memory": {
            "total_gb":     round(vm.total    / 1e9, 1),
            "used_gb":      round((vm.total - vm.available) / 1e9, 1),
            "available_gb": round(vm.available / 1e9, 1),
            "pressure":     _memory_pressure(),
        },
        "processes": procs[:n_procs],
        "catalog": {
            "stt": STT_CATALOG,
            "tts": TTS_CATALOG,
            "llm": LLM_CATALOG,
        },
    }
