"""Instance-segmentation quality metrics (object matching by IoU)."""
from __future__ import annotations

import numpy as np


def match(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one matching of predicted objects to ground-truth objects by IoU.

    Returns tp/fp/fn, precision, recall, f1, ap (= tp / (tp+fp+fn), the Cellpose/StarDist "AP")
    and the mean IoU of the matched pairs."""
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch {gt.shape} vs {pred.shape}")
    gt_ids = np.unique(gt[gt > 0])
    pr_ids = np.unique(pred[pred > 0])
    tp, ious = 0, []
    if len(gt_ids) and len(pr_ids):
        both = (gt > 0) & (pred > 0)
        base = np.int64(pred.max()) + 1
        codes, inter = np.unique(gt[both].astype(np.int64) * base + pred[both], return_counts=True)
        g, p = codes // base, codes % base
        g_area = dict(zip(*np.unique(gt[gt > 0], return_counts=True)))
        p_area = dict(zip(*np.unique(pred[pred > 0], return_counts=True)))
        iou = inter / (np.array([g_area[a] + p_area[b] for a, b in zip(g, p)], float) - inter)
        used_g, used_p = set(), set()
        for k in np.argsort(-iou):
            if iou[k] < iou_threshold:
                break
            if g[k] in used_g or p[k] in used_p:
                continue
            used_g.add(g[k])
            used_p.add(p[k])
            ious.append(float(iou[k]))
        tp = len(ious)
    fp, fn = len(pr_ids) - tp, len(gt_ids) - tp
    return _summary(tp, fp, fn, ious)


def _summary(tp: int, fp: int, fn: int, ious: list[float]) -> dict:
    precision = tp / (tp + fp) if tp + fp else 1.0 if fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4), "ap": round(ap, 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0, "_ious": ious}


def combine(results: list[dict]) -> dict:
    """Pool several per-image results (sums counts, so big images weigh more)."""
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    ious = [i for r in results for i in r.get("_ious", [])]
    return _summary(tp, fp, fn, ious)


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}
