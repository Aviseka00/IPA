"""PyTorch datasets: full-image segmentation and ROI crops."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

from image_io import letterbox, read_image_rgb
from seg_io import instance_mask_to_binary, load_instance_mask


class SegmentationPairDataset(Dataset):
    """Letterboxed RGB + binary infected mask."""

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        image_size: int,
        augment: bool,
        seed: int = 42,
    ):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.train_tf = (
            A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.06,
                        scale_limit=0.15,
                        rotate_limit=45,
                        border_mode=0,
                        p=0.65,
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.55),
                    A.GaussNoise(std_range=(0.02, 0.08), p=0.35),
                ]
            )
            if augment
            else None
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, sp = self.pairs[idx]
        img = read_image_rgb(ip)
        inst = load_instance_mask(sp)
        if inst.shape[:2] != img.shape[:2]:
            import cv2

            inst = cv2.resize(inst, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = instance_mask_to_binary(inst)
        if self.train_tf is not None:
            r = self.train_tf(image=img, mask=mask)
            img, mask = r["image"], r["mask"]
        img, mask, _ = letterbox(img, mask, self.image_size, 0, 0)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return img_t, mask_t


def build_pairs(images_dir: Path | None = None) -> list[tuple[Path, Path]]:
    from discover_pairs import list_pairs

    if images_dir is not None:
        # local override for tests
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        pairs = []
        for p in sorted(Path(images_dir).iterdir()):
            if p.suffix.lower() in exts:
                sp = p.with_name(p.stem + "_seg.npy")
                if sp.exists():
                    pairs.append((p, sp))
        return pairs
    return list_pairs()
