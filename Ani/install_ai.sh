#!/usr/bin/env bash
# Add AI to Ani: Cellpose (auto-label), micro-sam (AI click), YOLO (your trained models). CPU only, internet once.
set -e
cd "$(dirname "$0")"
. .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname)" = "Linux" ]; then
  # the default Linux torch wheel brings ~2 GB of GPU libraries; the CPU build is much smaller
  python -m pip install "torch>=2.5" torchvision --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install "torch>=2.5" torchvision
fi
python -m pip install "cellpose>=3.1,<4" ultralytics
if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
  python -m pip install micro_sam "mobile_sam @ https://github.com/ChaoningZhang/MobileSAM/archive/refs/heads/master.zip"
else
  echo "micro-sam skipped: needs Python 3.11+"
fi
python -m ani --download-models || true
python -m ani --selftest
echo "=== Done. In Ani open the AI tab and press Refresh. ==="
