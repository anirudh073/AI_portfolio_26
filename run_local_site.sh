#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$ROOT_DIR/docs"
SITE_PORT="${SITE_PORT:-8000}"
DEFAULT_INFERENCE_URL="${DEFAULT_INFERENCE_URL:-http://localhost:8787}"

if [[ ! -d "$DOCS_DIR" ]]; then
  echo "Could not find docs directory at $DOCS_DIR" >&2
  exit 1
fi

INFERENCE_URL="${1:-}"

if [[ -z "$INFERENCE_URL" ]]; then
  read -r -p "Inference URL [$DEFAULT_INFERENCE_URL]: " INFERENCE_URL
  INFERENCE_URL="${INFERENCE_URL:-$DEFAULT_INFERENCE_URL}"
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to run the local site." >&2
  exit 1
fi

ENCODED_INFERENCE_URL="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$INFERENCE_URL")"
SITE_URL="http://localhost:${SITE_PORT}/?inference_url=${ENCODED_INFERENCE_URL}"

echo "Serving BrainBlast locally from $DOCS_DIR"
echo "Inference URL: $INFERENCE_URL"
echo "Open: $SITE_URL"

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$SITE_URL" >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
  open "$SITE_URL" >/dev/null 2>&1 &
fi

cd "$DOCS_DIR"
exec python3 -m http.server "$SITE_PORT"
