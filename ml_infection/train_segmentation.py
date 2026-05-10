"""
Train binary segmentation U-Net for infected-cell regions.
Splits paired Images/*.jpg|png|tif + *_seg.npy into train/val/test by image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SegmentationPairDataset, build_pairs
from losses import BCEDiceLoss
from metrics_report import segmentation_metrics
from model_unet import UNet
from split_utils import split_indices


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()
    probs_all = []
    masks_all = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy()
        probs_all.append(p)
        masks_all.append(y.numpy())
    prob = np.concatenate(probs_all, axis=0)
    gt = np.concatenate(masks_all, axis=0)
    m = segmentation_metrics(prob[:, 0], gt[:, 0], threshold=threshold)
    return m, prob, gt


def plot_cm(cm: list[list[float]], out_path: Path, title: str):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        np.asarray(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.images_dir)
    if len(pairs) < 3:
        raise SystemExit("Need at least 3 image/seg pairs in Images/.")

    train_i, val_i, test_i = split_indices(len(pairs), args.seed)
    train_pairs = [pairs[i] for i in train_i]
    val_pairs = [pairs[i] for i in val_i]
    test_pairs = [pairs[i] for i in test_i]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SegmentationPairDataset(train_pairs, args.image_size, augment=True, seed=args.seed)
    val_ds = SegmentationPairDataset(val_pairs, args.image_size, augment=False, seed=args.seed)
    test_ds = SegmentationPairDataset(test_pairs, args.image_size, augment=False, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNet(in_ch=3, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
    crit = BCEDiceLoss(0.5)

    best_score = -1.0
    best_path = out_dir / "unet_infected_best.pt"

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            losses.append(loss.item())

        val_m, _, _ = evaluate(model, val_loader, device, args.threshold)
        sched.step(val_m["dice_hard"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val": val_m,
            }
        )
        print(
            f"epoch {epoch} loss={np.mean(losses):.4f} "
            f"val_dice={val_m['dice_hard']:.4f} prec={val_m['precision']:.4f} "
            f"rec={val_m['recall']:.4f} f1={val_m['f1']:.4f} auroc={val_m['auroc']:.4f}"
        )
        if val_m["dice_hard"] > best_score:
            best_score = val_m["dice_hard"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "image_size": args.image_size,
                    "threshold": args.threshold,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_m, _, _ = evaluate(model, test_loader, device, args.threshold)
    val_m, val_prob, val_gt = evaluate(model, val_loader, device, args.threshold)

    plot_cm(val_m["confusion_matrix"], out_dir / "confusion_matrix_val.png", "Validation (pixel-level)")
    plot_cm(test_m["confusion_matrix"], out_dir / "confusion_matrix_test.png", "Test (pixel-level)")

    report = {
        "split_sizes": {
            "train": len(train_pairs),
            "val": len(val_pairs),
            "test": len(test_pairs),
        },
        "segmentation_val": {k: v for k, v in val_m.items() if k != "confusion_matrix"},
        "segmentation_val_confusion_matrix": val_m["confusion_matrix"],
        "segmentation_test": {k: v for k, v in test_m.items() if k != "confusion_matrix"},
        "segmentation_test_confusion_matrix": test_m["confusion_matrix"],
        "best_checkpoint": str(best_path),
        "note": (
            "Pixel-level metrics on letterboxed %dx%d inputs. "
            "AUROC uses a subsample of pixels for speed. "
            "Clinical '99%%' targets usually require more images and domain validation."
            % (args.image_size, args.image_size)
        ),
    }
    with open(out_dir / "qc_segmentation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Export TorchScript for inference without source tree
    model.eval()
    example = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(out_dir / "unet_infected_traced.pt"))

    print("Saved:", best_path)
    print("QC:", json.dumps({k: report[k] for k in ["segmentation_val", "segmentation_test"]}, indent=2))


if __name__ == "__main__":
    main()
