#!/usr/bin/env bash
# ---- Ani one-time setup for Linux / macOS (needs internet only this once) ----
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null; then
  echo "Python 3.10+ not found. Install it (Ubuntu: sudo apt install python3 python3-venv)."; exit 1
fi
if [ "$(uname)" = "Linux" ] && ! ldconfig -p 2>/dev/null | grep -q libxcb-cursor; then
  echo "Note: Qt6 needs libxcb-cursor0 on Linux. If Ani fails to open, run:"
  echo "    sudo apt install libxcb-cursor0 python3-venv"
fi
"$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
chmod +x run_ani.sh
echo
echo "=== Ani is installed. Start it with ./run_ani.sh  (or ./run_ani.sh --demo to practise) ==="
