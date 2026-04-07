#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
DOCS_DIR="$ROOT_DIR/docs"
SITE_PORT="${SITE_PORT:-8000}"
INFERENCE_PORT="${INFERENCE_PORT:-8787}"
INFERENCE_URL="http://localhost:${INFERENCE_PORT}"
PREFERRED_CONDA_ENV="${PREFERRED_CONDA_ENV:-ai_portfolio}"
PYTHON_BIN=""

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to open BrainBlast locally." >&2
  exit 1
fi

if [[ ! -d "$DOCS_DIR" ]]; then
  echo "Could not find docs directory at $DOCS_DIR" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SITE_PID:-}" ]] && kill -0 "$SITE_PID" >/dev/null 2>&1; then
    kill "$SITE_PID" >/dev/null 2>&1 || true
  fi

  if [[ -n "${MODEL_PID:-}" ]] && kill -0 "$MODEL_PID" >/dev/null 2>&1; then
    kill "$MODEL_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

ensure_venv() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating local virtual environment..."
    python3 -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="$VENV_DIR/bin/python"
}

ensure_requirements() {
  if "$PYTHON_BIN" -c "import flask, flask_cors, torch, sentencepiece" >/dev/null 2>&1; then
    return
  fi

  echo "Installing Python dependencies from requirements_serve.txt..."
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements_serve.txt"
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts="${3:-120}"

  for ((i = 1; i <= attempts; i += 1)); do
    if python - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=1) as response:
    sys.exit(0 if 200 <= response.status < 500 else 1)
PY
    then
      return 0
    fi

    sleep 1
  done

  echo "$label did not become ready in time." >&2
  return 1
}

encode_url() {
  python -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"
}

activate_conda_env() {
  local env_name="$1"

  if ! command -v conda >/dev/null 2>&1; then
    return 1
  fi

  local conda_base
  conda_base="$(conda info --base 2>/dev/null)" || return 1

  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$env_name" >/dev/null 2>&1 || return 1
  PYTHON_BIN="$(command -v python)"
}

select_python() {
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    PYTHON_BIN="$(command -v python)"
    if "$PYTHON_BIN" -c "import flask, flask_cors, torch, sentencepiece" >/dev/null 2>&1; then
      echo "Using active Conda environment: ${CONDA_DEFAULT_ENV}"
      return
    fi
  fi

  if activate_conda_env "$PREFERRED_CONDA_ENV"; then
    if "$PYTHON_BIN" -c "import flask, flask_cors, torch, sentencepiece" >/dev/null 2>&1; then
      echo "Using Conda environment: $PREFERRED_CONDA_ENV"
      return
    fi
  fi

  ensure_venv
}

open_in_browser() {
  local url="$1"

  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 &
  elif command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 &
  fi
}

select_python
ensure_requirements

echo "Starting BrainBlast inference server on $INFERENCE_URL"
cd "$ROOT_DIR"
"$PYTHON_BIN" serve.py &
MODEL_PID=$!

wait_for_http "$INFERENCE_URL/health" "Inference server"

echo "Starting BrainBlast site on http://localhost:${SITE_PORT}"
cd "$DOCS_DIR"
python -m http.server "$SITE_PORT" >/dev/null 2>&1 &
SITE_PID=$!

wait_for_http "http://localhost:${SITE_PORT}/" "Static site"

SITE_URL="http://localhost:${SITE_PORT}/?inference_url=$(encode_url "$INFERENCE_URL")"
echo "Opening $SITE_URL"
open_in_browser "$SITE_URL"

echo "BrainBlast is running. Press Ctrl+C to stop both servers."
wait "$MODEL_PID"
