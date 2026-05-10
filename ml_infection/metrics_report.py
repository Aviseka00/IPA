"""QC metrics for binary segmentation and optional ROI classification."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def flatten_probs_labels(
    logits_or_prob: np.ndarray,
    targets: np.ndarray,
    max_pixels: int = 400_000,
    rng: np.random.Generator | None = None,
    from_logits: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten spatial arrays; optionally subsample for ROC stability."""
    if logits_or_prob.ndim > 1:
        p = logits_or_prob.reshape(-1)
    else:
        p = logits_or_prob
    t = targets.reshape(-1).astype(np.int32)
    if from_logits:
        import torch

        p = torch.as_tensor(p).sigmoid().numpy().astype(np.float64)
    else:
        p = np.clip(p.astype(np.float64), 0.0, 1.0)
    if len(p) > max_pixels:
        rng = rng or np.random.default_rng(42)
        idx = rng.choice(len(p), size=max_pixels, replace=False)
        p = p[idx]
        t = t[idx]
    return p, t


def soft_dice(prob: np.ndarray, gt: np.ndarray, threshold: float = 0.5, eps: float = 1e-6) -> float:
    p = (prob >= threshold).astype(np.float64).reshape(-1)
    t = (gt > 0).astype(np.float64).reshape(-1)
    inter = (p * t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


def segmentation_metrics(prob: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:
    """Pixel-wise binary metrics (probabilities vs 0/1 mask)."""
    pred = (prob >= threshold).astype(np.int32)
    g = (gt > 0).astype(np.int32).reshape(-1)
    pbin = pred.reshape(-1)
    acc = accuracy_score(g, pbin)
    prec = precision_score(g, pbin, zero_division=0)
    rec = recall_score(g, pbin, zero_division=0)
    f1 = f1_score(g, pbin, zero_division=0)
    cm = confusion_matrix(g, pbin, labels=[0, 1])
    pf, tf = flatten_probs_labels(prob, gt)
    try:
        roc = roc_auc_score(tf, pf)
    except ValueError:
        roc = float("nan")
    dice_hard = soft_dice(prob, gt, threshold=threshold)
    out = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "dice_hard": float(dice_hard),
        "auroc": float(roc),
        "confusion_matrix": cm.tolist(),
    }
    return out


def roi_classification_metrics(y_true: Iterable[int], y_score: Iterable[float], threshold: float = 0.5) -> dict:
    y_true = np.asarray(list(y_true), dtype=np.int32)
    s = np.asarray(list(y_score), dtype=np.float64)
    y_pred = (s >= threshold).astype(np.int32)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        roc = roc_auc_score(y_true, s)
    except ValueError:
        roc = float("nan")
    # PR-AUC
    p, r, _ = precision_recall_curve(y_true, s)
    pr_auc = auc(r, p) if len(r) > 1 else float("nan")
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": float(roc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
    }
