#!/usr/bin/env bash
set -euo pipefail
# Auto-setup venv and run app with locked deps (Linux/macOS)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt
python -m pip install -e .[ui]

python - <<'PY'
import pmdarima, numpy
PY

exec streamlit run app/AFMo.py
