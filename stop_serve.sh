#!/usr/bin/env bash
# Stops the background inference server started by serve_background.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/.serve.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found — server may not be running."
  exit 0
fi

PID="$(cat "$PID_FILE")"

if kill -0 "$PID" >/dev/null 2>&1; then
  kill "$PID"
  echo "Inference server (PID $PID) stopped."
else
  echo "No process found for PID $PID — already stopped."
fi

rm -f "$PID_FILE"
