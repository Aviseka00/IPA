"""Train segmentation U-Net, ROI classifier, then merge QC JSON."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-epochs", type=int, default=60)
    ap.add_argument("--roi-epochs", type=int, default=45)
    ap.add_argument("--skip-roi", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    py = sys.executable
    r = subprocess.run(
        [py, str(here / "train_segmentation.py"), "--epochs", str(args.seg_epochs)],
        cwd=str(here.parent),
    )
    if r.returncode != 0:
        sys.exit(r.returncode)
    if not args.skip_roi:
        r = subprocess.run(
            [py, str(here / "roi_train.py"), "--epochs", str(args.roi_epochs)],
            cwd=str(here.parent),
        )
        if r.returncode != 0:
            sys.exit(r.returncode)
    r = subprocess.run([py, str(here / "merge_qc.py")], cwd=str(here.parent))
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
