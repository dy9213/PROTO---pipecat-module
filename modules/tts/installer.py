"""
Download and install the VOICEVOX Engine binary for Apple Silicon (macOS arm64).
Files are placed in modules/tts/bin/ alongside bundled resources.
"""
import hashlib, io, json, os, shutil, socket, stat, subprocess, tarfile, time, urllib.request, zipfile
from pathlib import Path
from typing import Callable, Optional

BIN_DIR    = Path(__file__).parent / "bin"
GITHUB_API = "https://api.github.com/repos/VOICEVOX/voicevox_engine/releases/latest"

ProgressCb = Callable[[int, str], None]

_BIN_CANDIDATES = ["run", "voicevox_engine", "main"]


def is_installed() -> bool:
    for name in _BIN_CANDIDATES:
        p = BIN_DIR / name
        if p.exists() and os.access(p, os.X_OK):
            return True
    return False


_RETRYABLE = (socket.gaierror, ConnectionResetError, TimeoutError, urllib.error.URLError)

def _urlopen_retry(url: str, timeout: int, retries: int = 3, delay: float = 2.0):
    req = urllib.request.Request(url, headers={"User-Agent": "OniChat/1.0"})
    last_err = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except _RETRYABLE as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_err

def _fetch_json(url: str) -> dict:
    with _urlopen_retry(url, timeout=15) as r:
        return json.loads(r.read())


def _fetch_checksums(assets: list) -> dict[str, str]:
    """Return {filename: sha256} from a checksums asset if one exists."""
    asset = next(
        (a for a in assets if "sha256" in a["name"].lower() or "checksum" in a["name"].lower()),
        None,
    )
    if asset is None:
        return {}
    try:
        req = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": "OniChat/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            text = r.read().decode()
        result = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                sha, fname = parts
                result[fname.lstrip("*")] = sha.lower()
        return result
    except Exception:
        return {}


def install(progress: Optional[ProgressCb] = None) -> None:
    def report(pct: int, msg: str):
        if progress:
            progress(pct, msg)

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = BIN_DIR / ".tmp-install"

    report(0, "Fetching latest VOICEVOX Engine release…")
    release = _fetch_json(GITHUB_API)
    tag     = release.get("tag_name", "unknown")
    assets  = release["assets"]

    # Prefer .vvpp (zip) over .7z.001 to avoid 7z extraction dependency
    def _is_macos_arm64(name):
        n = name.lower()
        return "macos" in n and any(a in n for a in ["arm64", "aarch64"])

    asset = (
        next((a for a in assets if _is_macos_arm64(a["name"]) and a["name"].endswith(".vvpp")), None)
        or next((a for a in assets if _is_macos_arm64(a["name"]) and a["name"].endswith((".zip", ".tar.gz", ".tgz"))), None)
    )
    if asset is None:
        raise RuntimeError(
            f"No macOS arm64 build found in VOICEVOX Engine release {tag}.\n"
            "Please download manually from:\n"
            "  https://github.com/VOICEVOX/voicevox_engine/releases\n"
            f"and place the engine binary in: {BIN_DIR}"
        )

    report(3, "Fetching checksums…")
    checksums    = _fetch_checksums(assets)
    expected_sha = checksums.get(asset["name"])

    url     = asset["browser_download_url"]
    size    = asset.get("size", 0)
    size_mb = size // 1_000_000
    name    = asset["name"]
    report(5, f"Downloading {name} ({size_mb} MB) — {tag}")

    buf    = io.BytesIO()
    hasher = hashlib.sha256()
    with _urlopen_retry(url, timeout=300) as r:
        downloaded = 0
        while True:
            chunk = r.read(65_536)
            if not chunk:
                break
            buf.write(chunk)
            hasher.update(chunk)
            downloaded += len(chunk)
            if size:
                pct = 5 + int(downloaded / size * 74)
                report(pct, f"Downloading… {downloaded // 1_000_000}/{size_mb} MB")

    if expected_sha:
        report(80, "Verifying integrity…")
        actual_sha = hasher.hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"SHA-256 mismatch — archive may be corrupt or tampered.\n"
                f"  expected: {expected_sha}\n"
                f"  got:      {actual_sha}"
            )

    report(82, "Extracting…")
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir()
        buf.seek(0)

        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            with tarfile.open(fileobj=buf, mode="r:gz") as tf:
                tf.extractall(tmp_dir)
        elif name.endswith((".zip", ".vvpp")):
            # Write to a temp file and use system unzip — Python's zipfile
            # doesn't restore symlinks (writes target path as plain text instead).
            tmp_zip = tmp_dir.parent / (name + ".tmp")
            try:
                tmp_zip.write_bytes(buf.getvalue())
                result = subprocess.run(
                    ["unzip", "-q", str(tmp_zip), "-d", str(tmp_dir)],
                    capture_output=True, text=True,
                )
                if result.returncode not in (0, 1):   # 1 = warnings only
                    raise RuntimeError(f"unzip failed: {result.stderr.strip()}")
            finally:
                tmp_zip.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Unsupported archive format: {name}")

        # Make extracted binaries executable
        for f in tmp_dir.rglob("*"):
            if f.is_file() and not f.suffix and not f.name.startswith("."):
                f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # Verify binary present before committing
        found = any((tmp_dir / c).exists() for c in _BIN_CANDIDATES)
        if not found:
            # Check one level deep (archive may have a top-level folder)
            for sub in tmp_dir.iterdir():
                if sub.is_dir():
                    if any((sub / c).exists() for c in _BIN_CANDIDATES):
                        # Promote contents of sub-folder
                        for item in sub.iterdir():
                            shutil.move(str(item), tmp_dir / item.name)
                        sub.rmdir()
                        found = True
                        break

        if not any((tmp_dir / c).exists() for c in _BIN_CANDIDATES):
            contents = [f.name for f in tmp_dir.iterdir()]
            raise RuntimeError(
                f"Engine binary not found after extraction. "
                f"Expected one of {_BIN_CANDIDATES}. Found: {contents}"
            )

        report(95, "Installing…")
        for f in tmp_dir.iterdir():
            dest = BIN_DIR / f.name
            if dest.exists():
                shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
            shutil.move(str(f), dest)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    report(100, f"Installed VOICEVOX Engine {tag}")
