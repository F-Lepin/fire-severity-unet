"""One-hot encoding for categorical LULC rasters."""

from __future__ import annotations

import numpy as np


def lulc_to_onehot(lulc: np.ndarray, class_ids: list[int]) -> np.ndarray:
    """
    Convert a 2D LULC array into a one-hot stack (C, H, W).

    Pixels with values not in class_ids produce all-zero channels.
    """
    h, w = lulc.shape
    n_classes = len(class_ids)
    onehot = np.zeros((n_classes, h, w), dtype=np.float32)
    for idx, class_id in enumerate(class_ids):
        onehot[idx] = (lulc == class_id).astype(np.float32)
    return onehot


def onehot_argmax(onehot: np.ndarray, class_ids: list[int]) -> np.ndarray:
    """Recover discrete LULC from a one-hot stack."""
    indices = onehot.argmax(axis=0)
    out = np.zeros(onehot.shape[1:], dtype=np.int16)
    for idx, class_id in enumerate(class_ids):
        out[indices == idx] = class_id
    return out


def remap_severity(
    severity: np.ndarray,
    class_map: dict[int, int],
    nodata_values: set[int] | None = None,
) -> np.ndarray:
    """
    Map raw severity values to model classes (0=ignore, 1-3=severity levels).
    """
    out = np.zeros_like(severity, dtype=np.int64)
    nodata_values = nodata_values or set()
    for raw, mapped in class_map.items():
        out[severity == raw] = mapped
    if nodata_values:
        for nd in nodata_values:
            out[severity == nd] = 0
    return out


def build_loss_mask(scar: np.ndarray, severity: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True where loss should be computed (inside scar, valid severity).
    """
    inside_scar = scar > 0
    valid_severity = severity > 0
    return inside_scar & valid_severity
