#!/usr/bin/env bash
# Starts the BrainBlast inference server in the background.
# Survives terminal close. Runs until the machine reboots or you run stop_serve.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
INFERENCE_PORT="${INFERENCE_PORT:-8787}"
PID_FILE="$ROOT_DIR/.serve.pid"
LOG_FILE="$ROOT_DIR/.serve.log"

# ── Pick Python ────────────────────────────────────────────────────────────────
PYTHON_BIN=""

# 1. Active conda env
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

# 2. Preferred conda env
if [[ -z "$PYTHON_BIN" ]] && command -v conda >/dev/null 2>&1; then
  PREFERRED="${PREFERRED_CONDA_ENV:-ai_portfolio}"
  CONDA_BASE="$(conda info --base 2>/dev/null)" || true
  if [[ -n "$CONDA_BASE" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$PREFERRED" >/dev/null 2>&1 || true
    PYTHON_BIN="$(command -v python)"
  fi
fi

# 3. Local venv (created by open_brainblast.sh on first run)
if [[ -z "$PYTHON_BIN" ]] || ! "$PYTHON_BIN" -c "import flask, flask_cors, torch, sentencepiece" >/dev/null 2>&1; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "No Python environment found with the required packages."
    echo "Run ./open_brainblast.sh once first (then Ctrl+C), which sets up the venv."
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

# ── Guard against double-start ─────────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "Inference server is already running (PID $OLD_PID)."
    echo "Open GitHub Pages with:"
    echo "  https://anirudh073.github.io/AI_portfolio_26/?inference_url=http://localhost:${INFERENCE_PORT}"
    exit 0
  fi
fi

# ── Start ──────────────────────────────────────────────────────────────────────
cd "$ROOT_DIR"
nohup "$PYTHON_BIN" serve.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Inference server started (PID $(cat "$PID_FILE"))."
echo "Logs: $LOG_FILE"
echo ""
echo "Open GitHub Pages with:"
echo "  https://anirudh073.github.io/AI_portfolio_26/?inference_url=http://localhost:${INFERENCE_PORT}"
echo ""
echo "To stop the server: ./stop_serve.sh"
