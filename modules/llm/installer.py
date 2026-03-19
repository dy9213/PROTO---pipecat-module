"""
Download and install llama-server binary for Apple Silicon from llama.cpp releases.
"""
import hashlib, io, json, os, shutil, socket, stat, tarfile, time, urllib.request, zipfile
from pathlib import Path
from typing import Callable, Optional

BIN_DIR    = Path(__file__).parent / "bin"
LLAMA_BIN  = BIN_DIR / "llama-server"
GITHUB_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

ProgressCb = Callable[[int, str], None]


def is_installed() -> bool:
    return LLAMA_BIN.exists() and os.access(LLAMA_BIN, os.X_OK)


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

    report(0, "Fetching latest llama.cpp release…")
    release  = _fetch_json(GITHUB_API)
    tag      = release.get("tag_name", "unknown")
    assets   = release["assets"]

    asset = next(
        (a for a in assets
         if "macos-arm64" in a["name"]
         and a["name"].endswith((".zip", ".tar.gz", ".tgz"))),
        None,
    )
    if asset is None:
        raise RuntimeError("No macOS arm64 binary found in latest llama.cpp release")

    report(3, "Fetching checksums…")
    checksums = _fetch_checksums(assets)
    expected_sha = checksums.get(asset["name"])

    url     = asset["browser_download_url"]
    size    = asset.get("size", 0)
    size_mb = size // 1_000_000
    report(5, f"Downloading {asset['name']} ({size_mb} MB) — {tag}")

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
        name = asset["name"]
        buf.seek(0)
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            with tarfile.open(fileobj=buf, mode="r:gz") as tf:
                tf.extractall(tmp_dir)
            # flatten one level of subdirectory if present
            entries = list(tmp_dir.iterdir())
            if len(entries) == 1 and entries[0].is_dir():
                for f in entries[0].iterdir():
                    shutil.move(str(f), tmp_dir / f.name)
                entries[0].rmdir()
            for f in tmp_dir.rglob("*"):
                if f.is_file() and not f.suffix and not f.name.startswith("."):
                    f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        else:
            with zipfile.ZipFile(buf) as zf:
                for info in zf.infolist():
                    fname = Path(info.filename).name
                    if not fname or fname.startswith("."):
                        continue
                    dest = tmp_dir / fname
                    dest.write_bytes(zf.read(info.filename))
                    if not fname.endswith((".dylib", ".metallib", ".plist", ".txt", ".md")):
                        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        if not (tmp_dir / "llama-server").exists():
            raise RuntimeError("llama-server binary not found in archive after extraction")

        report(95, "Installing…")
        for f in tmp_dir.iterdir():
            dest = BIN_DIR / f.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(f), dest)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    report(100, f"Installed llama-server {tag}")
