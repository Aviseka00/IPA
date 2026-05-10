from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        t = targets.float()
        inter = (prob * t).sum(dim=(1, 2, 3))
        union = prob.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
        dice = (2 * inter + eps) / (union + eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE with class imbalance correction + soft Dice."""

    def __init__(self, bce_weight: float = 0.35):
        super().__init__()
        self.dice = DiceLoss()
        self.w = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.float()
        pos = t.sum().clamp(min=1.0)
        neg = t.numel() - t.sum()
        pw = float((neg / pos).clamp(1.0, 500.0).item())
        bce = F.binary_cross_entropy_with_logits(
            logits, t, pos_weight=logits.new_tensor(pw, dtype=logits.dtype, device=logits.device)
        )
        return self.w * bce + (1.0 - self.w) * self.dice(logits, targets)
