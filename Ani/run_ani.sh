#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "Ani is not installed yet. Run ./setup_linux_mac.sh first."
  exit 1
fi
export PYTHONUTF8=1
export QT_ENABLE_HIGHDPI_SCALING=1
exec .venv/bin/python -m ani "$@"
