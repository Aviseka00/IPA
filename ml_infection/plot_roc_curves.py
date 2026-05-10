"""Plot pixel-level ROC curves for train / val / test using the saved U-Net checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from datasets import SegmentationPairDataset, build_pairs
from metrics_report import flatten_probs_labels
from model_unet import UNet
from split_utils import split_indices


@torch.no_grad()
def collect_probs_gt(model, loader, device):
    model.eval()
    probs_all, gts_all = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu().numpy()
        probs_all.append(p)
        gts_all.append(y.numpy())
    prob = np.concatenate(probs_all, axis=0)
    gt = np.concatenate(gts_all, axis=0)
    return prob[:, 0], gt[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "unet_infected_best.pt",
    )
    ap.add_argument("--out", type=Path, default=None, help="PNG path (default: artifacts/roc_curves.png)")
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pixels", type=int, default=400_000, help="Subsample pixels per split for ROC speed")
    args = ap.parse_args()

    out_path = args.out or (Path(__file__).resolve().parent / "artifacts" / "roc_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    imsz = int(ckpt.get("image_size", args.image_size))

    pairs = build_pairs(args.images_dir)
    train_i, val_i, test_i = split_indices(len(pairs), args.seed)
    train_pairs = [pairs[i] for i in train_i]
    val_pairs = [pairs[i] for i in val_i]
    test_pairs = [pairs[i] for i in test_i]

    train_ds = SegmentationPairDataset(train_pairs, imsz, augment=False, seed=args.seed)
    val_ds = SegmentationPairDataset(val_pairs, imsz, augment=False, seed=args.seed)
    test_ds = SegmentationPairDataset(test_pairs, imsz, augment=False, seed=args.seed)

    loader_kw = dict(batch_size=1, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_ds, **loader_kw)
    val_loader = DataLoader(val_ds, **loader_kw)
    test_loader = DataLoader(test_ds, **loader_kw)

    model = UNet(in_ch=3, base=32).to(device)
    model.load_state_dict(ckpt["model_state"])

    splits = [
        ("Train", train_loader, 42),
        ("Validation", val_loader, 43),
        ("Test", test_loader, 44),
    ]

    plt.figure(figsize=(8, 7))
    for name, loader, rng_seed in splits:
        prob, gt = collect_probs_gt(model, loader, device)
        rng = np.random.default_rng(rng_seed)
        p_flat, t_flat = flatten_probs_labels(prob, gt, max_pixels=args.max_pixels, rng=rng)
        if np.unique(t_flat).size < 2:
            continue
        fpr, tpr, _ = roc_curve(t_flat, p_flat)
        roc_auc = roc_auc_score(t_flat, p_flat)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUROC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC — pixel-level infected vs background (subsampled)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved", out_path.resolve())


if __name__ == "__main__":
    main()
