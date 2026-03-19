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
    {"key": "qwen3-1.7b-4bit", "label": "Qwen3-ASR 1.7B 4bit",   "size_gb": 0.6},
    {"key": "qwen3-1.7b-8bit", "label": "Qwen3-ASR 1.7B 8bit",   "size_gb": 1.8},
    {"key": "qwen3-1.7b-bf16", "label": "Qwen3-ASR 1.7B bf16",   "size_gb": 3.4},
    {"key": "qwen3-0.6b-4bit", "label": "Qwen3-ASR 0.6B 4bit",   "size_gb": 0.3},
    {"key": "qwen3-0.6b-8bit", "label": "Qwen3-ASR 0.6B 8bit",   "size_gb": 0.4},
    {"key": "qwen3-0.6b-bf16", "label": "Qwen3-ASR 0.6B bf16",   "size_gb": 0.8},
]

TTS_CATALOG = [
    {"key": "voicevox", "label": "VoiceVox (Japanese Only)", "size_gb": 0.4, "file_gb": 1.7},
]

# size_gb = runtime memory (weights + KV cache @ 4k ctx + compute buffers)
# file_gb = download size (weights only)
LLM_CATALOG = [
    {"key": "qwen3.5-2b-q4km",  "label": "Qwen3.5  2B Q4_K_M",     "size_gb": 1.8,  "file_gb": 1.33},
    {"key": "qwen3.5-4b-q4km",  "label": "Qwen3.5  4B Q4_K_M",     "size_gb": 3.7,  "file_gb": 2.87},
    {"key": "qwen3.5-9b-q4km",  "label": "Qwen3.5  9B Q4_K_M",     "size_gb": 7.1,  "file_gb": 5.89},
    {"key": "qwen3.5-27b-q4km", "label": "Qwen3.5 27B Q4_K_M",     "size_gb": 19.0, "file_gb": 17.13},
    {"key": "qwen3.5-35b-q4km", "label": "Qwen3.5 35B-A3B Q4_K_M", "size_gb": 23.0, "file_gb": 21.40},
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
        return {0: "normal", 1: "normal", 2: "warn", 4: "critical"}.get(val, "unknown")
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
            mem = p.info.get("memory_info")
            if mem is None:
                continue
            rss_gb = mem.rss / 1e9
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
