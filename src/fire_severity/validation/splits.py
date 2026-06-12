"""Fire-level train/validation splits to avoid spatial leakage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


def fire_id_from_path(path: Path) -> str:
    return path.stem.replace("_patches", "")


def group_patch_files_by_fire(patch_files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in patch_files:
        fid = fire_id_from_path(path)
        groups.setdefault(fid, []).append(path)
    return groups


def leave_one_fire_out_splits(
    patch_files: list[Path],
) -> list[tuple[list[Path], list[Path], str]]:
    """Return (train_files, val_files, held_out_fire_id) for each fire."""
    groups = group_patch_files_by_fire(patch_files)
    splits = []
    for held_out in sorted(groups):
        val_files = groups[held_out]
        train_files = [f for fid, files in groups.items() if fid != held_out for f in files]
        splits.append((train_files, val_files, held_out))
    return splits


def holdout_fire_splits(
    patch_files: list[Path],
    val_fraction: float = 0.30,
    seed: int = 42,
) -> list[tuple[list[Path], list[Path], list[str], list[str]]]:
    """
    Single random holdout: ``val_fraction`` of fires go to validation (entire scars).

    Returns one split: (train_files, val_files, train_fire_ids, val_fire_ids).
    """
    groups = group_patch_files_by_fire(patch_files)
    fire_ids = sorted(groups.keys())
    n_fires = len(fire_ids)
    if n_fires < 2:
        raise ValueError("Need at least 2 fires for holdout split.")

    n_val = max(1, int(round(n_fires * val_fraction)))
    n_val = min(n_val, n_fires - 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_fires)
    val_set = {fire_ids[i] for i in perm[:n_val]}

    train_files: list[Path] = []
    val_files: list[Path] = []
    for fid in fire_ids:
        if fid in val_set:
            val_files.extend(groups[fid])
        else:
            train_files.extend(groups[fid])

    train_fires = sorted(fid for fid in fire_ids if fid not in val_set)
    val_fires = sorted(val_set)
    return [(train_files, val_files, train_fires, val_fires)]


def kfold_fire_splits(
    patch_files: list[Path],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[Path], list[Path], list[str]]]:
    groups = group_patch_files_by_fire(patch_files)
    fire_ids = sorted(groups)
    file_list: list[Path] = []
    group_ids: list[str] = []
    for fid in fire_ids:
        for f in groups[fid]:
            file_list.append(f)
            group_ids.append(fid)

    gkf = GroupKFold(n_splits=min(n_folds, len(fire_ids)))
    splits = []
    for train_idx, val_idx in gkf.split(file_list, groups=group_ids):
        train_files = [file_list[i] for i in train_idx]
        val_files = [file_list[i] for i in val_idx]
        val_fires = sorted({group_ids[i] for i in val_idx})
        splits.append((train_files, val_files, val_fires))
    return splits


def confusion_matrix_masked(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss_mask: np.ndarray,
    num_classes: int = 4,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = loss_mask & (y_true > 0)
    for t, p, m in zip(y_true[valid], y_pred[valid], valid[valid]):
        cm[int(t), int(p)] += 1
    return cm
