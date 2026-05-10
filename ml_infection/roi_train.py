"""
ROI-level classifier (ResNet18): infected cell crops vs background crops.
Provides patch-level precision/recall/F1/AUROC/confusion matrix (QC complement).
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm

from image_io import read_image_rgb
from metrics_report import roi_classification_metrics
from seg_io import load_instance_mask


def crop_safe(img: np.ndarray, cy: int, cx: int, s: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    y0, x0 = cy - s // 2, cx - s // 2
    y1, x1 = y0 + s, x0 + s
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return None
    return img[y0:y1, x0:x1]


def extract_roi_tensors(
    pairs: list[tuple[Path, Path]],
    crop_size: int,
    negatives_per_positive: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import cv2

    rng = np.random.default_rng(seed)
    xs = []
    ys = []

    for ip, sp in pairs:
        img = read_image_rgb(ip)
        inst = load_instance_mask(sp)
        if inst.shape[:2] != img.shape[:2]:
            inst = cv2.resize(inst, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        binm = inst > 0
        ids = [int(i) for i in np.unique(inst) if i > 0]
        for lab in ids:
            ys_, xs_ = np.where(inst == lab)
            cy = int(np.mean(ys_))
            cx = int(np.mean(xs_))
            patch = crop_safe(img, cy, cx, crop_size)
            if patch is None:
                continue
            xs.append(patch)
            ys.append(1)
        neg_needed = len(ids) * negatives_per_positive
        bg_attempts = 0
        picked = 0
        h, w = binm.shape
        while picked < neg_needed and bg_attempts < neg_needed * 80:
            bg_attempts += 1
            cy = rng.integers(crop_size // 2, h - crop_size // 2)
            cx = rng.integers(crop_size // 2, w - crop_size // 2)
            patch = crop_safe(img, cy, cx, crop_size)
            if patch is None:
                continue
            sl = binm[cy - crop_size // 2 : cy - crop_size // 2 + crop_size, cx - crop_size // 2 : cx - crop_size // 2 + crop_size]
            if sl.sum() == 0:
                xs.append(patch)
                ys.append(0)
                picked += 1

    if not xs:
        raise RuntimeError("No ROI patches extracted — check masks and crop_size.")

    batch = np.stack([cv2.resize(x, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR) for x in xs])
    xt = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
    yt = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    return xt, yt


class SmallROIClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=w)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x)


def plot_cm(cm: list[list[float]], out_path: Path, title: str):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        np.asarray(cm),
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Pred bg", "Pred infected"],
        yticklabels=["True bg", "True infected"],
    )
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def split_lists(pairs, seed):
    from split_utils import split_indices

    n = len(pairs)
    tr, va, te = split_indices(n, seed)
    return [pairs[i] for i in tr], [pairs[i] for i in va], [pairs[i] for i in te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import build_pairs

    pairs = build_pairs(args.images_dir)
    train_p, val_p, test_p = split_lists(pairs, args.seed)

    X_tr, y_tr = extract_roi_tensors(train_p, args.crop_size, args.neg_per_pos, args.seed)
    X_va, y_va = extract_roi_tensors(val_p, args.crop_size, args.neg_per_pos, args.seed + 1)
    X_te, y_te = extract_roi_tensors(test_p, args.crop_size, args.neg_per_pos, args.seed + 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False)

    model = SmallROIClassifier(pretrained=True).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_path = out_dir / "roi_resnet18_best.pt"
    best_f1 = -1.0

    def run_epoch(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        losses = []
        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, leave=False):
                x, y = x.to(device), y.to(device)
                if train:
                    opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = crit(logits, y)
                if train:
                    loss.backward()
                    opt.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def scores_labels(loader):
        model.eval()
        scores = []
        labels = []
        for x, y in loader:
            x = x.to(device)
            lg = model(x).squeeze(1)
            prob = torch.sigmoid(lg).cpu().numpy()
            scores.extend(list(prob))
            labels.extend(list(y.squeeze(1).numpy().astype(int)))
        return labels, scores

    history = []
    for epoch in range(1, args.epochs + 1):
        tl = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        y_true, y_score = scores_labels(val_loader)
        m = roi_classification_metrics(y_true, y_score, threshold=0.5)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val": m})
        print(f"epoch {epoch} train_loss={tl:.4f} val_loss={vl:.4f} val_f1={m['f1']:.4f} val_auroc={m['auroc']:.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"model_state": model.state_dict(), "crop_size": args.crop_size}, best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    y_va, s_va = scores_labels(val_loader)
    y_te, s_te = scores_labels(test_loader)
    m_va = roi_classification_metrics(y_va, s_va, 0.5)
    m_te = roi_classification_metrics(y_te, s_te, 0.5)

    plot_cm(m_va["confusion_matrix"], out_dir / "confusion_matrix_roi_val.png", "ROI validation")
    plot_cm(m_te["confusion_matrix"], out_dir / "confusion_matrix_roi_test.png", "ROI test")

    report = {
        "roi_val": {k: v for k, v in m_va.items() if k != "confusion_matrix"},
        "roi_val_confusion_matrix": m_va["confusion_matrix"],
        "roi_test": {k: v for k, v in m_te.items() if k != "confusion_matrix"},
        "roi_test_confusion_matrix": m_te["confusion_matrix"],
        "checkpoint": str(best_path),
        "counts": {"train_rois": len(X_tr), "val_rois": len(X_va), "test_rois": len(X_te)},
    }
    with open(out_dir / "qc_roi.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "roi_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    model.eval()
    ex = torch.randn(1, 3, args.crop_size, args.crop_size, device=device)
    traced = torch.jit.trace(model, ex)
    traced.save(str(out_dir / "roi_resnet18_traced.pt"))
    print("Saved ROI model:", best_path)


if __name__ == "__main__":
    main()
