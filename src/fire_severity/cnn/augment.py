"""Spatial augmentations for one-hot LULC patches (train only)."""

from __future__ import annotations

import random

import torch


def augment_lulc_onehot(x: torch.Tensor, cfg: dict, rng: random.Random | None = None) -> torch.Tensor:
    """
    Apply random flips / 90° rotations to a (C, H, W) one-hot patch.

    Preserves categorical structure; only valid for geometric transforms.
    """
    if not cfg.get("enabled", False):
        return x

    rng = rng or random

    if cfg.get("horizontal_flip", True) and rng.random() < 0.5:
        x = torch.flip(x, dims=[2])
    if cfg.get("vertical_flip", True) and rng.random() < 0.5:
        x = torch.flip(x, dims=[1])
    if cfg.get("rotate90", True) and rng.random() < 0.5:
        k = rng.randint(1, 3)
        x = torch.rot90(x, k, dims=[1, 2])
    return x
