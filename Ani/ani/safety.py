"""Crash-safe files, a local recovery copy, and a log for each installation.

Ani runs on the user's own computer, on the CPU. A graphics card is never required.
"""
from __future__ import annotations

import atexit
import faulthandler
import hashlib
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np

_LOG = logging.getLogger("ani")
_FAULT_STREAMS: list = []
_LOCK_HELD: Path | None = None


def _limits() -> tuple[int, int, int]:
    from .machine import hardware

    info = hardware()
    return int(info["workers"]), int(info["feature_pixels"]), int(info["feature_cells"])


def prepare_cpu() -> int:
    """Fit maths threads to this PC and leave cores free so the window stays smooth.

    Existing environment values are kept, so a user can still choose a thread count.
    Returns the worker count Ani will use when the variables were not already set.
    """
    workers, feature_pixels, feature_cells = _limits()
    global MAX_FEATURE_PIXELS, MAX_FEATURE_CELLS
    MAX_FEATURE_PIXELS = feature_pixels
    MAX_FEATURE_CELLS = feature_cells
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, str(workers))
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(os.environ["OMP_NUM_THREADS"]))
    except Exception:
        pass
    _warm_maths()
    return workers


def _warm_maths() -> None:
    """Prime SciPy filters once so the first Label now is not a cold start."""
    try:
        import numpy as np
        from scipy import ndimage as ndi

        probe = np.zeros((64, 64), np.float32)
        ndi.uniform_filter(probe, size=11)
        ndi.minimum_filter(probe, size=5)
        ndi.gaussian_filter1d(probe[0], sigma=1.0)
    except Exception:
        pass


# Defaults for imports that run before prepare_cpu(); prepare_cpu refreshes them.
MAX_FEATURE_CELLS = 10_000
MAX_FEATURE_PIXELS = 25_000_000


def data_dir() -> Path:
    """Per-user folder for logs and unsaved-work copies."""
    override = os.environ.get("ANI_DATA_DIR")
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    root = Path(local) if local else Path.home() / ".local" / "share"
    return root / "Ani"


def install_logging() -> None:
    """Write errors to a rotating file. Safe to call more than once."""
    if _LOG.handlers:
        return
    folder = data_dir() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(folder / "ani.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _LOG.setLevel(logging.INFO)
    _LOG.addHandler(handler)
    _LOG.propagate = False
    fault = folder / "fault.log"
    stream = fault.open("a", encoding="utf-8")
    faulthandler.enable(stream)
    _FAULT_STREAMS.append(stream)


def install_excepthook() -> None:
    """Record an uncaught error, then keep the normal Python report."""
    install_logging()
    previous = sys.excepthook

    def _hook(exc_type, exc, tb):
        _LOG.error("unhandled error", exc_info=(exc_type, exc, tb))
        previous(exc_type, exc, tb)

    sys.excepthook = _hook


def log_error(where: str, exc: BaseException) -> None:
    install_logging()
    _LOG.error("%s: %s", where, exc, exc_info=exc)


def atomic_write(path, write) -> Path:
    """Write to a temporary file, then replace `path`. A failure leaves the old file in place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.partial{path.suffix}")
    try:
        write(temporary)
        try:
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def feature_job_error(height: int, width: int, n_cells: int) -> str | None:
    """Refuse one measurement that would exhaust memory. Returns the reason, or None when it can run."""
    pixels = int(height) * int(width)
    if pixels > MAX_FEATURE_PIXELS:
        return (
            f"This image has {pixels:,} pixels. Feature extraction stops at "
            f"{MAX_FEATURE_PIXELS:,} so the computer does not run out of memory."
        )
    if n_cells > MAX_FEATURE_CELLS:
        return (
            f"{n_cells:,} cells is above the {MAX_FEATURE_CELLS:,} cell limit for one extraction."
        )
    return None


def save_recovery(image_path, masks, class_names, classes, root=None) -> Path:
    """Keep a crash copy of the outlines for this image."""
    folder = Path(root) if root is not None else data_dir() / "recovery"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{_image_key(image_path)}.npz"

    def write(path: Path) -> None:
        np.savez_compressed(
            path,
            masks=np.asarray(masks),
            names=np.array(json.dumps(list(class_names))),
            classes=np.array(json.dumps({str(int(k)): str(v) for k, v in classes.items()})),
            image=np.array(str(image_path)),
        )

    return atomic_write(target, write)


def load_recovery(image_path, root=None) -> dict | None:
    """Return a crash copy, or None when there is none or it cannot be read."""
    path = (Path(root) if root is not None else data_dir() / "recovery") / f"{_image_key(image_path)}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {
                "masks": np.asarray(data["masks"]),
                "names": json.loads(str(data["names"].item())),
                "classes": {int(k): str(v) for k, v in json.loads(str(data["classes"].item())).items()},
                "image": str(data["image"].item()),
            }
    except Exception as exc:
        log_error(f"recovery {path.name}", exc)
        return None


def clear_recovery(image_path, root=None) -> None:
    path = (Path(root) if root is not None else data_dir() / "recovery") / f"{_image_key(image_path)}.npz"
    path.unlink(missing_ok=True)


def remember_image(path, root=None) -> None:
    """Keep a short list of images this person opened."""
    if root is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return
    folder = Path(root) if root is not None else data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "recent.json"
    text = str(Path(path).resolve())
    items = [text] + [item for item in recent_images(root) if item != text]
    payload = json.dumps(items[:12], indent=2)
    atomic_write(target, lambda file: file.write_text(payload, encoding="utf-8"))


def recent_images(root=None) -> list[str]:
    folder = Path(root) if root is not None else data_dir()
    target = folder / "recent.json"
    if not target.is_file():
        return []
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [str(item) for item in loaded if item][:12]


def acquire_instance(root=None) -> bool:
    """One Ani window per computer. A second double-click finds the one already open."""
    global _LOCK_HELD
    folder = Path(root) if root is not None else data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "ani.lock"
    if lock.is_file():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = 0
        if pid and pid != os.getpid() and _process_alive(pid):
            return False
    lock.write_text(str(os.getpid()), encoding="utf-8")
    _LOCK_HELD = lock
    atexit.register(release_instance, root)
    return True


def release_instance(root=None) -> None:
    folder = Path(root) if root is not None else data_dir()
    lock = folder / "ani.lock"
    try:
        if lock.is_file() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _image_key(image_path) -> str:
    text = str(Path(image_path).resolve()).lower().encode()
    return hashlib.sha256(text).hexdigest()[:20]
