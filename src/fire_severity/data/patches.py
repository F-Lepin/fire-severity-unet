"""Spatial patch extraction with burn-scar-centered sampling criteria."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fire_severity.data.encoding import build_loss_mask, lulc_to_onehot, remap_severity


@dataclass
class PatchCriteria:
    size: int = 32
    min_burn_fraction: float = 0.25
    min_valid_severity_fraction: float = 0.20
    max_outside_scar_fraction: float = 0.75
    samples_per_fire: int = 200
    balance_severity: bool = True
    random_seed: int = 42
    severity_classes: tuple[int, ...] = (1, 2, 3, 4)


@dataclass
class PatchSample:
    fire_id: str
    row: int
    col: int
    center_severity: int
    burn_fraction: float
    valid_severity_fraction: float


def _half(size: int) -> int:
    return size // 2


def extract_window(arr: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    """Extract a size×size window centered at (row, col), zero-padded at edges."""
    half = _half(size)
    r0, r1 = row - half, row + half
    c0, c1 = col - half, col + half
    h, w = arr.shape[-2], arr.shape[-1]
    out_shape = (size, size) if arr.ndim == 2 else (arr.shape[0], size, size)
    out = np.zeros(out_shape, dtype=arr.dtype)

    src_r0, src_r1 = max(r0, 0), min(r1, h)
    src_c0, src_c1 = max(c0, 0), min(c1, w)
    dst_r0 = src_r0 - r0
    dst_c0 = src_c0 - c0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    if arr.ndim == 2:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    else:
        out[:, dst_r0:dst_r1, dst_c0:dst_c1] = arr[:, src_r0:src_r1, src_c0:src_c1]
    return out


def evaluate_center(
    scar: np.ndarray,
    severity: np.ndarray,
    row: int,
    col: int,
    criteria: PatchCriteria,
) -> PatchSample | None:
    """Check whether a window centered at (row, col) meets selection criteria."""
    size = criteria.size
    half = _half(size)

    if scar[row, col] <= 0:
        return None

    scar_win = extract_window(scar, row, col, size)
    sev_win = extract_window(severity, row, col, size)
    n = size * size

    burn_fraction = float((scar_win > 0).sum()) / n
    valid_fraction = float((sev_win > 0).sum()) / n
    outside_fraction = 1.0 - burn_fraction

    if burn_fraction < criteria.min_burn_fraction:
        return None
    if valid_fraction < criteria.min_valid_severity_fraction:
        return None
    if outside_fraction > criteria.max_outside_scar_fraction:
        return None

    center_sev = int(severity[row, col])
    if center_sev <= 0:
        return None

    return PatchSample(
        fire_id="",
        row=row,
        col=col,
        center_severity=center_sev,
        burn_fraction=burn_fraction,
        valid_severity_fraction=valid_fraction,
    )


def candidate_centers(scar: np.ndarray, severity: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of (row, col) inside scar with valid severity."""
    mask = (scar > 0) & (severity > 0)
    rows, cols = np.where(mask)
    return np.stack([rows, cols], axis=1)


def select_patches(
    fire_id: str,
    scar: np.ndarray,
    severity: np.ndarray,
    criteria: PatchCriteria,
) -> list[PatchSample]:
    """
    Select patch centers for one fire, optionally balancing by severity class.
    """
    rng = np.random.default_rng(criteria.random_seed + hash(fire_id) % 10000)
    centers = candidate_centers(scar, severity)
    if len(centers) == 0:
        return []

    accepted: list[PatchSample] = []
    by_class: dict[int, list[PatchSample]] = {c: [] for c in criteria.severity_classes}

    order = rng.permutation(len(centers))
    for idx in order:
        row, col = int(centers[idx, 0]), int(centers[idx, 1])
        sample = evaluate_center(scar, severity, row, col, criteria)
        if sample is None:
            continue
        sample.fire_id = fire_id
        by_class.setdefault(sample.center_severity, []).append(sample)
        accepted.append(sample)

    if not criteria.balance_severity:
        if len(accepted) <= criteria.samples_per_fire:
            return accepted
        indices = rng.choice(len(accepted), criteria.samples_per_fire, replace=False)
        return [accepted[i] for i in indices]

    n_classes = len(criteria.severity_classes)
    per_class = max(1, criteria.samples_per_fire // n_classes)
    balanced: list[PatchSample] = []
    for sev_class in criteria.severity_classes:
        pool = by_class.get(sev_class, [])
        if not pool:
            continue
        n = min(len(pool), per_class)
        pick = rng.choice(len(pool), n, replace=False)
        balanced.extend(pool[i] for i in pick)

    if len(balanced) < criteria.samples_per_fire:
        remaining = [s for s in accepted if s not in balanced]
        need = criteria.samples_per_fire - len(balanced)
        if remaining and need > 0:
            extra = rng.choice(len(remaining), min(need, len(remaining)), replace=False)
            balanced.extend(remaining[i] for i in extra)

    return balanced[: criteria.samples_per_fire]


def build_patch_arrays(
    lulc: np.ndarray,
    severity: np.ndarray,
    scar: np.ndarray,
    samples: list[PatchSample],
    class_ids: list[int],
    severity_class_map: dict[int, int],
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build X (N,C,H,W), Y (N,H,W), loss_mask (N,H,W), meta arrays.
    """
    n = len(samples)
    n_classes = len(class_ids)
    x = np.zeros((n, n_classes, patch_size, patch_size), dtype=np.float32)
    y = np.zeros((n, patch_size, patch_size), dtype=np.int64)
    loss_mask = np.zeros((n, patch_size, patch_size), dtype=bool)

    severity_mapped = remap_severity(severity, severity_class_map)
    lulc_onehot_full = lulc_to_onehot(lulc, class_ids)

    for i, sample in enumerate(samples):
        x[i] = extract_window(lulc_onehot_full, sample.row, sample.col, patch_size)
        y[i] = extract_window(severity_mapped, sample.row, sample.col, patch_size)
        scar_win = extract_window(scar, sample.row, sample.col, patch_size)
        loss_mask[i] = build_loss_mask(scar_win, y[i])

    return x, y, loss_mask, np.array([s.center_severity for s in samples])


def save_patch_bundle(
    output_path: Path,
    fire_id: str,
    x: np.ndarray,
    y: np.ndarray,
    loss_mask: np.ndarray,
    samples: list[PatchSample],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(
        {
            "fire_id": [s.fire_id for s in samples],
            "row": [s.row for s in samples],
            "col": [s.col for s in samples],
            "center_severity": [s.center_severity for s in samples],
            "burn_fraction": [s.burn_fraction for s in samples],
            "valid_severity_fraction": [s.valid_severity_fraction for s in samples],
        }
    )
    np.savez_compressed(
        output_path,
        fire_id=fire_id,
        x=x,
        y=y,
        loss_mask=loss_mask,
        meta=meta.to_dict(orient="list"),
    )


def load_patch_bundle(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
