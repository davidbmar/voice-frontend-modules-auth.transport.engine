#!/usr/bin/env bash
# First-time setup for voice-frontend development and testing.
# Usage: ./scripts/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== voice-frontend setup ==="
echo

# 1. Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment exists."
fi

source .venv/bin/activate
echo "Python: $(python --version) at $(which python)"
echo

# 2. Upgrade pip
pip install --upgrade pip --quiet

# 3. Install the three packages in editable mode
echo "Installing packages (editable)..."
pip install -e packages/transport --quiet
pip install -e packages/edge-auth --quiet
pip install -e packages/engine-starter --quiet
echo "  transport, edge-auth, engine-starter: installed"

# 4. Install dev dependencies
echo "Installing dev dependencies..."
pip install pytest pytest-asyncio ruff --quiet
echo "  pytest, pytest-asyncio, ruff: installed"

# 5. Install engine runtime deps (STT, TTS)
echo "Installing STT/TTS runtime deps..."
pip install faster-whisper piper-tts --quiet
echo "  faster-whisper, piper-tts: installed"

# 5b. Install optional Kokoro TTS (fails gracefully if onnxruntime unavailable)
echo "Installing Kokoro TTS (optional)..."
if pip install kokoro-onnx --quiet 2>/dev/null; then
    echo "  kokoro-onnx: installed"
    echo "  Note: download model files to ~/.cache/voice-frontend/kokoro/"
    echo "    kokoro-v1.0.onnx + voices.bin from https://github.com/thewh1teagle/kokoro-onnx/releases"
else
    echo "  kokoro-onnx: skipped (install manually if needed)"
fi

# 6. Install server
echo "Installing server..."
pip install 'uvicorn[standard]' --quiet
echo "  uvicorn: installed"

# 7. Check for Ollama
echo
echo "=== Checking external services ==="
if command -v ollama &>/dev/null; then
    echo "Ollama: installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
    if ollama list 2>/dev/null | grep -q "qwen3"; then
        echo "  qwen3 model: available"
    else
        echo "  No qwen3 model found. To install: ollama pull qwen3:8b"
    fi
else
    echo "Ollama: not installed"
    echo "  Install: brew install ollama && ollama pull qwen3:8b"
    echo "  Or set ANTHROPIC_API_KEY / OPENAI_API_KEY to use a cloud LLM instead"
fi

# 8. Check for cloudflared (optional, only needed for remote access)
if command -v cloudflared &>/dev/null; then
    echo "cloudflared: installed"
else
    echo "cloudflared: not installed (optional — only needed for remote/mobile testing)"
fi

# 9. Check for Twilio (optional)
if [ -n "${TWILIO_ACCOUNT_SID:-}" ] && [ -n "${TWILIO_AUTH_TOKEN:-}" ]; then
    echo "Twilio: credentials set"
else
    echo "Twilio: not configured (optional — localhost works without TURN servers)"
fi

# 10. Run tests
echo
echo "=== Running tests ==="
python -m pytest packages/transport/tests packages/edge-auth/tests packages/engine-starter/tests -v --tb=short

echo
echo "=== Setup complete ==="
echo
echo "To run:"
echo "  source .venv/bin/activate"
echo "  ./scripts/run.sh                # Interactive mode selection"
echo "  ./scripts/run.sh --local        # Local (localhost only)"
echo "  ./scripts/run.sh --lan          # LAN (phone on same WiFi)"
echo "  ./scripts/run.sh --tunnel       # Cellular (Cloudflare Tunnel)"
echo "  ./scripts/run.sh --check        # Validate config only"
echo
