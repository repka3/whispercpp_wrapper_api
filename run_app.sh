#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7600}"

if [[ ! -x ".venv/bin/uvicorn" ]]; then
  echo "Missing .venv/bin/uvicorn. Create the venv and install requirements first:" >&2
  echo "  ./install.sh" >&2
  exit 1
fi

if [[ ! -f ".env" ]]; then
  echo "Missing .env. Copy .env.example to .env and set WHISPERCPP_BASE_DIR." >&2
  exit 1
fi

exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
