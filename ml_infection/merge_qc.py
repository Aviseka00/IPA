"""Combine qc_segmentation.json + qc_roi.json into qc_summary.json."""
from __future__ import annotations

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "artifacts"
    seg = root / "qc_segmentation.json"
    roi = root / "qc_roi.json"
    out = root / "qc_summary.json"
    data = {}
    if seg.exists():
        data["segmentation"] = json.loads(seg.read_text(encoding="utf-8"))
    if roi.exists():
        data["roi_classifier"] = json.loads(roi.read_text(encoding="utf-8"))
    if not data:
        raise SystemExit("No qc_segmentation.json or qc_roi.json found in artifacts/.")
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
