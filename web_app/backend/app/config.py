"""Application settings."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

WEB_APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WEB_APP_ROOT.parent
BACKEND_ROOT = WEB_APP_ROOT / "backend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", extra="ignore")

    app_name: str = "IPA Cell Analysis Platform"
    app_version: str = "1.1.0-part11"
    api_prefix: str = "/api"
    secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    # Part 11: shorter session; idle handled client-side + short JWT
    access_token_expire_minutes: int = 30
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "ipa_analysis"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    storage_root: Path = WEB_APP_ROOT / "storage"
    repo_root: Path = REPO_ROOT
    ipa4_config: Path = REPO_ROOT / "ipa4" / "config_ipa4.yaml"
    inference_device: str = "cpu"
    inference_conf: float = 0.25
    max_upload_mb: int = 50
    max_files_per_batch: int = 5000
    # Multi-user / team scale (shared API; Ani stays per-desktop)
    uvicorn_workers: int = 4
    inference_max_concurrent: int = 2  # heavy GPU/CPU jobs; queue the rest
    mongodb_max_pool_size: int = 200
    mongodb_min_pool_size: int = 10
    api_request_timeout_s: int = 600
    # EVOS tiled-mosaic restitch before detection (standalone preprocessing)
    mosaic_restitch_enabled: bool = True
    mosaic_restitch_on_low_confidence: str = "passthrough_warn"  # or "use_restitched"
    # Shared reset code for /auth/reset-password (set in .env; leave empty to disable self-service reset)
    password_reset_token: str = ""

    # --- 21 CFR Part 11 / GAMP 5 technical controls ---
    part11_mode: bool = True
    password_min_length: int = 8
    password_require_upper: bool = True
    password_require_lower: bool = True
    password_require_digit: bool = True
    password_require_special: bool = True
    password_history_count: int = 5
    max_failed_logins: int = 5
    lockout_minutes: int = 15
    require_esignature_on_publish: bool = True
    allow_open_registration: bool = False
    record_retention_years: int = 7
    gamp_category: str = "Category 4 (Configured Product) with Category 5 AI/ML components"

    # Ani v0.4.0 human-label bridge (vendored at repo_root/Ani)
    ani_root: Path = REPO_ROOT / "Ani"
    ani_python: str = ""  # empty → use ani_root/.venv/Scripts/python.exe
    retrain_dataset_dir: Path = REPO_ROOT / "datasets" / "retrain_from_review"
    # When True, Ani starts automatically whenever the IPA backend (or app) starts
    ani_autostart: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def ensure_dirs() -> None:
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    Path(settings.retrain_dataset_dir).mkdir(parents=True, exist_ok=True)
    (Path(settings.retrain_dataset_dir) / "images").mkdir(parents=True, exist_ok=True)
    (Path(settings.retrain_dataset_dir) / "labels").mkdir(parents=True, exist_ok=True)
