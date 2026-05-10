"""Paths and MongoDB settings."""
from __future__ import annotations

import os
from pathlib import Path

# web_app/ directory (parent of backend/)
WEB_APP_ROOT = Path(__file__).resolve().parents[2]
IPA3_ROOT = WEB_APP_ROOT.parent

UPLOAD_DIR = WEB_APP_ROOT / "uploads" / "test_images"
PROCESSED_DIR = WEB_APP_ROOT / "processed_images"
MODEL_CHECKPOINT = IPA3_ROOT / "ml_infection" / "artifacts" / "unet_infected_best.pt"

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.environ.get("MONGODB_DB", "cell_infection_analysis")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
