"""Discover raw image + _seg.npy pairs under Images/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "Images"


def seg_path_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.stem + "_seg.npy")


def list_pairs():
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    pairs = []
    for p in sorted(IMG_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        sp = seg_path_for_image(p)
        if sp.exists():
            pairs.append((p, sp))
    return pairs


if __name__ == "__main__":
    ps = list_pairs()
    print("pairs", len(ps))
    for a, b in ps[:5]:
        print(a.name, "->", b.name)
