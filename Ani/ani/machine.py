"""Match Ani's maths threads and limits to this computer."""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def hardware() -> dict:
    """CPU and RAM snapshot for this PC."""
    logical = int(os.cpu_count() or 2)
    ram_gb = _ram_gb()
    # Leave two logical processors for Qt and Windows so drawing stays smooth.
    workers = max(1, min(6, logical - 2 if logical > 2 else 1))
    # With ~16 GB, keep feature jobs under half the RAM for image buffers.
    if ram_gb >= 14:
        # ~16 GB: room for a 2k×2k RGB scan plus outlines, without crowding Windows.
        feature_pixels = 30_000_000
        feature_cells = 12_000
    elif ram_gb >= 8:
        feature_pixels = 25_000_000
        feature_cells = 10_000
    else:
        feature_pixels = 12_000_000
        feature_cells = 5_000
        workers = min(workers, 3)
    return {
        "logical": logical,
        "ram_gb": ram_gb,
        "workers": workers,
        "feature_pixels": feature_pixels,
        "feature_cells": feature_cells,
    }


def _ram_gb() -> float:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullTotalPhys) / (1024 ** 3)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PAGE_SIZE")
        count = os.sysconf("SC_PHYS_PAGES")
        return float(pages * count) / (1024 ** 3)
    except Exception:
        return 8.0


def summary() -> str:
    info = hardware()
    return (
        f"This PC: {info['logical']} CPU threads, {info['ram_gb']:.0f} GB RAM. "
        f"Ani uses {info['workers']} maths threads and leaves the rest for a smooth window."
    )
