#!/bin/bash
set -e

# Requires Python 3.10+ and ffmpeg installed via Homebrew:
#   brew install ffmpeg
# Uses whatever python3 is available (3.10+ is fine)
# Note: Silero VAD model (~2MB) is downloaded automatically on first app start.

PYTHON=$(which python3.12 2>/dev/null || which python3.11 2>/dev/null || which python3.10 2>/dev/null || which python3 2>/dev/null || echo "")
if [ -z "$PYTHON" ]; then
  echo "Error: python3 not found."
  exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
  echo "Error: ffmpeg not found. Run: brew install ffmpeg"
  exit 1
fi

$PYTHON -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install duckduckgo-search

# Install mlx-audio on Apple Silicon only
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ] && [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Apple Silicon detected — installing mlx-audio and mlx-whisper for local TTS/STT…"
  pip install mlx-audio
  pip install mlx-whisper
else
  echo "Skipping mlx-audio / mlx-whisper (Apple Silicon only)"
fi

mkdir -p data
echo "Bootstrap complete. Run: npm start"
