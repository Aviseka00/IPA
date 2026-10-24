"""IPA + Ani robustness / parameter suite before git push.

Run from repo root:
  web_app\\backend\\.venv\\Scripts\\python.exe scripts\\robustness_test.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "web_app" / "backend"
sys.path.insert(0, str(BACKEND))

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print("=== IPA + Ani robustness suite ===\n")

    print("[1] Imports & config")
    from app.config import ensure_dirs, settings
    from app.main import app
    from app.services import ani_bridge

    ensure_dirs()
    check("ani bundled", (REPO / "Ani" / "pyproject.toml").is_file())
    check("ani resolve", ani_bridge.resolve_ani_root() == (REPO / "Ani").resolve())
    check("ani version", ani_bridge.ani_version() not in ("", "unknown"), ani_bridge.ani_version())
    check("ani venv", (ani_bridge.resolve_ani_root() / ".venv" / "Scripts" / "python.exe").is_file())
    check("run_ani.bat", (REPO / "run_ani.bat").is_file())
    check("run_ipa.bat", (REPO / "run_ipa.bat").is_file())
    check("autostart default", settings.ani_autostart is True)
    check("inference limit", settings.inference_max_concurrent >= 1)
    check("mongo pool", settings.mongodb_max_pool_size >= 50)

    print("\n[2] API routes present")
    paths = {getattr(r, "path", "") for r in app.routes}
    for p in (
        "/api/health",
        "/api/ready",
        "/api/ani/status",
        "/api/ani/ensure-running",
        "/api/analysis/batches/{batch_id}/ani/open",
        "/api/analysis/batches/{batch_id}/ani/import",
        "/api/analysis/batches/{batch_id}/ani/status",
    ):
        check(p, p in paths)

    print("\n[3] Mask / GT logic")
    import numpy as np

    dets = [
        {"id": "a", "x": 0.3, "y": 0.3, "class_id": 0, "confidence": 0.9, "box": [0.2, 0.2, 0.4, 0.4]},
        {"id": "b", "x": 0.7, "y": 0.7, "class_id": 0, "confidence": 0.8, "box": [0.6, 0.6, 0.8, 0.8]},
        {"id": "junk", "x": 0.1, "y": 0.1, "class_id": 1, "confidence": 0.5, "box": [0.05, 0.05, 0.1, 0.1]},
    ]
    mask, classes = ani_bridge.detections_to_mask(dets, 128, 128)
    check("infected instances", int(mask.max()) == 2, str(int(mask.max())))
    check("count infected", ani_bridge.count_infected(mask, classes) == 2)
    gt = mask.copy()
    gt[5:20, 5:20] = 9
    classes[9] = "Infected"
    stats = ani_bridge.compare_masks(mask, gt)
    check("FN added", stats["added"] == 1, str(stats))
    check("matched kept", stats["matched"] == 2, str(stats))
    # FP remove one pred
    gt2 = mask.copy()
    gt2[gt2 == 1] = 0
    stats2 = ani_bridge.compare_masks(mask, gt2)
    check("FP removed", stats2["removed"] == 1, str(stats2))

    print("\n[4] Ani project build + import")
    from app.services.storage import ensure_batch_dirs

    emp, batch = "robust_emp", f"robust_{int(time.time())}"
    dirs = ensure_batch_dirs(emp, batch)
    src = REPO / "test" / "QS_3075.jpg"
    if not src.is_file():
        src = next((REPO / "test").glob("*.jpg"), None)
    check("test image", src is not None and src.is_file(), str(src))
    if src and src.is_file():
        shutil.copy2(src, dirs["input"] / src.name)
        items = [
            {
                "item_id": f"{batch}_{src.name}",
                "filename": src.name,
                "status": "completed",
                "input_path": str(dirs["input"] / src.name),
                "ai_infected_count": 2,
                "infected_count": 2,
                "ai_detections": dets[:2],
                "model_id": "ipa8",
            }
        ]
        root = ani_bridge.build_ani_project(emp, batch, items)
        check("project.json", (root / "project.json").is_file())
        check("prediction mask", (root / "predictions" / f"{src.stem}.tif").is_file())
        lab = root / "labels" / "cells"
        lab.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "predictions" / f"{src.stem}.tif", lab / f"{src.stem}.tif")
        shutil.copy2(root / "predictions" / f"{src.stem}.classes.json", lab / f"{src.stem}.classes.json")
        # concurrent manifest appends
        def one_import(i: int):
            it = dict(items[0])
            it["item_id"] = f"{batch}_parallel_{i}"
            # unique stem via copy
            stem = f"{src.stem}_p{i}"
            shutil.copy2(dirs["input"] / src.name, dirs["input"] / f"{stem}.jpg")
            it["filename"] = f"{stem}.jpg"
            it["input_path"] = str(dirs["input"] / f"{stem}.jpg")
            # rebuild single-image project bits
            h, w = ani_bridge._image_hw(Path(it["input_path"]))
            m, cl = ani_bridge.detections_to_mask(dets[:2], h, w)
            ani_bridge._write_mask_tif(root / "predictions" / f"{stem}.tif", m)
            ani_bridge._write_classes_json(root / "predictions" / f"{stem}.classes.json", cl)
            shutil.copy2(root / "images" / src.name, root / "images" / f"{stem}.jpg")
            ani_bridge._write_mask_tif(lab / f"{stem}.tif", m)
            ani_bridge._write_classes_json(lab / f"{stem}.classes.json", cl)
            return ani_bridge.import_ani_labels(emp, batch, [it], "ipa8")["imported"]

        # ensure original image also in ani images (already from build)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(one_import, i) for i in range(8)]
            results = [f.result() for f in as_completed(futs)]
        check("parallel imports", all(r == 1 for r in results), str(results))
        man = Path(settings.retrain_dataset_dir) / "manifest.jsonl"
        check("manifest exists", man.is_file())
        if man.is_file():
            lines = [ln for ln in man.read_text(encoding="utf-8").splitlines() if ln.strip()]
            check("manifest parseable", all(json.loads(ln) for ln in lines[-20:]), f"{len(lines)} lines")

    print("\n[5] Live HTTP (if server up)")
    try:
        import urllib.request

        for url in (
            "http://127.0.0.1:8010/api/health",
            "http://127.0.0.1:8010/api/ready",
            "http://127.0.0.1:8010/api/ani/status",
        ):
            with urllib.request.urlopen(url, timeout=8) as resp:
                body = json.loads(resp.read().decode())
                check(url, resp.status == 200, json.dumps(body)[:120])
    except Exception as e:
        print(f"  SKIP live HTTP — {e}")

    print("\n=== Summary ===")
    print(f"passed={passed} failed={failed}")
    if failed:
        print("NOT READY for git push")
        return 1
    print("READY for git push")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
