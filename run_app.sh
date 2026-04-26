#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7600}"

if [[ ! -x ".venv/bin/uvicorn" ]]; then
  echo "Missing .venv/bin/uvicorn. Create the venv and install requirements first:" >&2
  echo "  python -m venv .venv" >&2
  echo "  . .venv/bin/activate" >&2
  echo "  pip install -r requirements.txt" >&2
  exit 1
fi

exec .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" "$@"
