"""Loss functions with burn-scar masking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy that ignores pixels outside the burn scar (class 0).

    Uses nn.CrossEntropyLoss(ignore_index=0) plus optional extra masking
    for pixels flagged in loss_mask.
    """

    def __init__(self, ignore_index: int = 0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction="none",
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per_pixel = self.ce(logits, targets)
        if loss_mask is not None:
            per_pixel = per_pixel * loss_mask.float()
            denom = loss_mask.float().sum().clamp(min=1.0)
            return per_pixel.sum() / denom
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return per_pixel.sum() * 0.0
        return per_pixel[valid].mean()


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    """Inverse-frequency weights over valid pixels."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        if c == ignore_index:
            continue
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1.0)
    inv = 1.0 / counts
    inv[ignore_index] = 0.0
    weights = inv / inv[inv > 0].sum() * (inv > 0).sum()
    return weights
