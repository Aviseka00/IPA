"""
Run infection segmentation on a new microscopy image and estimate infected-cell count.

Uses letterboxing consistent with training and rescores the mask back to full resolution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from image_io import letterbox, read_image_rgb
from model_unet import UNet


class InfectionSegmenter:
    """Loads U-Net once; run ``predict()`` on many images."""

    def __init__(self, ckpt_path: Path, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(Path(ckpt_path), map_location=self.device, weights_only=False)
        self.image_size = int(ckpt.get("image_size", 512))
        self.default_threshold = float(ckpt.get("threshold", 0.5))
        self.model = UNet(in_ch=3, base=32).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_path: Path,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Returns binary mask (H,W), probability map (H,W), meta dict."""
        thr = self.default_threshold if threshold is None else float(threshold)
        img = read_image_rgb(Path(image_path))
        inst_dummy = np.zeros(img.shape[:2], dtype=np.uint8)
        lb_img, _, meta = letterbox(img, inst_dummy, self.image_size, 0, 0)

        xt = torch.from_numpy(lb_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        logits = self.model(xt)
        prob_small = torch.sigmoid(logits)[0, 0].cpu().numpy()

        prob_full = invert_letterbox_mask(prob_small, meta, (img.shape[0], img.shape[1]))
        bin_mask = (prob_full >= thr).astype(np.uint8)

        meta_out = {
            "threshold": thr,
            "image_size_train": self.image_size,
            "orig_shape_hw": [int(img.shape[0]), int(img.shape[1])],
        }
        return bin_mask, prob_full.astype(np.float32), meta_out

    @torch.no_grad()
    def predict_rgb(
        self,
        rgb: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Same as ``predict`` but accepts H×W×3 RGB uint8 (e.g. after CLAHE)."""
        thr = self.default_threshold if threshold is None else float(threshold)
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("rgb must be HxWx3 uint8")
        inst_dummy = np.zeros(img.shape[:2], dtype=np.uint8)
        lb_img, _, meta = letterbox(img, inst_dummy, self.image_size, 0, 0)

        xt = torch.from_numpy(lb_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        logits = self.model(xt)
        prob_small = torch.sigmoid(logits)[0, 0].cpu().numpy()

        prob_full = invert_letterbox_mask(prob_small, meta, (img.shape[0], img.shape[1]))
        bin_mask = (prob_full >= thr).astype(np.uint8)

        meta_out = {
            "threshold": thr,
            "image_size_train": self.image_size,
            "orig_shape_hw": [int(img.shape[0]), int(img.shape[1])],
        }
        return bin_mask, prob_full.astype(np.float32), meta_out


def invert_letterbox_mask(mask_small: np.ndarray, meta: dict, orig_hw: tuple[int, int]) -> np.ndarray:
    """Resize predicted mask from padded square back to original image size."""
    h, w = orig_hw
    scale = meta["scale"]
    dx, dy = meta["pad_x"], meta["pad_y"]
    nh = int(round(h * scale))
    nw = int(round(w * scale))
    crop = mask_small[dy : dy + nh, dx : dx + nw].astype(np.float32)
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def predict_mask(
    image_path: Path,
    ckpt_path: Path,
    device: torch.device | None = None,
    threshold: float | None = None,
    image_size: int | None = None,
) -> tuple[np.ndarray, dict]:
    seg = InfectionSegmenter(ckpt_path, device=device)
    if image_size is not None:
        seg.image_size = image_size
    bin_mask, _, meta_out = seg.predict(image_path, threshold=threshold)
    return bin_mask, meta_out


def count_instances(binary_mask: np.ndarray, min_area_px: int = 40) -> tuple[int, np.ndarray]:
    bin8 = (binary_mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin8, connectivity=8)
    count = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            count += 1
    return count, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="Microscopy image path")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "unet_infected_best.pt",
    )
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=40, help="Minimum CC area in pixels (full-res)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    mask, meta = predict_mask(args.image, args.checkpoint, threshold=args.threshold)
    n_cells, labels = count_instances(mask, min_area_px=args.min_area)

    out = {
        "image": str(args.image.resolve()),
        "infected_cell_count_estimate": int(n_cells),
        "meta": meta,
        "min_area_px": args.min_area,
    }
    print(json.dumps(out, indent=2))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
