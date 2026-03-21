#!/bin/bash
set -e

# Usage: bootstrap.sh [venv-destination]
#   venv-destination defaults to ./venv (dev mode)
#
# In production, Electron passes:
#   $1              = path to venv in userData
#   ONICHAT_UV      = path to bundled uv binary
#   ONICHAT_APP_ROOT = read-only app bundle root (cwd is set to this)

VENV_DIR="${1:-$(pwd)/venv}"

# ── Locate or download uv ─────────────────────────────────────────────────────
# Priority: bundled uv → system uv → download uv via curl
# Never falls back to system python3 — avoids triggering macOS Python installer.
if [ -n "$ONICHAT_UV" ] && [ -x "$ONICHAT_UV" ]; then
  UV="$ONICHAT_UV"
  echo "Using bundled uv: $UV"
elif command -v uv &>/dev/null; then
  UV="$(which uv)"
  echo "Using system uv: $UV"
else
  echo "uv not found — downloading to a temporary location…"
  UV_TMP_DIR="$(mktemp -d)"
  # uv's official installer writes the binary to ~/.cargo/bin by default;
  # instead we pull the binary directly for the current platform (macOS arm64).
  UV_BIN="$UV_TMP_DIR/uv"
  UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz"
  if ! curl -fsSL "$UV_URL" | tar -xz -C "$UV_TMP_DIR" --strip-components=1 2>/dev/null; then
    echo "Error: failed to download uv. Check your internet connection."
    rm -rf "$UV_TMP_DIR"
    exit 1
  fi
  UV="$UV_BIN"
  chmod +x "$UV"
  echo "Downloaded uv to $UV"
fi

# ── Create venv and install dependencies ─────────────────────────────────────
"$UV" venv --python 3.12 "$VENV_DIR"
"$UV" pip install --python "$VENV_DIR/bin/python" -r backend/requirements.txt
"$UV" pip install --python "$VENV_DIR/bin/python" mlx-audio

# Clean up temporary uv download if used
if [ -n "$UV_TMP_DIR" ]; then
  rm -rf "$UV_TMP_DIR"
fi

mkdir -p "$(dirname "$VENV_DIR")/data"
echo "Bootstrap complete."
