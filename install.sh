#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR=".venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: Python 3.12 is required, but '$PYTHON_BIN' was not found on PATH." >&2
  echo "Install Python 3.12, then rerun this script." >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys

raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)
PY
then
  version="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  echo "Error: Python 3.12 is required, but '$PYTHON_BIN' is Python $version." >&2
  exit 1
fi

if [[ -d "$VENV_DIR" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Error: '$VENV_DIR' already exists but does not contain an executable Python." >&2
    echo "Remove '$VENV_DIR' or recreate it with Python 3.12." >&2
    exit 1
  fi

  if ! "$VENV_DIR/bin/python" - <<'PY'
import sys

raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)
PY
  then
    version="$("$VENV_DIR/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    echo "Error: '$VENV_DIR' uses Python $version, but Python 3.12 is required." >&2
    echo "Remove '$VENV_DIR' and rerun this script." >&2
    exit 1
  fi
else
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "Install complete. Start the app with ./run_app.sh"
