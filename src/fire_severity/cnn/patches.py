"""Patch extraction for CNN LULC classifier (center-pixel binary labels)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fire_severity.data.encoding import lulc_to_onehot, remap_severity
from fire_severity.data.patches import PatchCriteria, evaluate_center, extract_window
from fire_severity.interpretability.landscape_metrics import class_proportions, summarize_patch


@dataclass
class CNNPatchCriteria:
    size: int = 32
    stride: int = 16
    min_burn_fraction: float = 0.25
    min_valid_severity_fraction: float = 0.20
    max_outside_scar_fraction: float = 0.75
    max_invalid_lulc_fraction: float = 0.10
    label_strategy: str = "center"
    random_seed: int = 42


def severity_to_binary_label(severity_class: int) -> int | None:
    """Map raster severity (1=low, 2=high) to model label (0=low, 1=high)."""
    if severity_class == 1:
        return 0
    if severity_class == 2:
        return 1
    return None


def dominant_severity_in_patch(
    severity: np.ndarray,
    scar: np.ndarray,
    row: int,
    col: int,
    size: int,
) -> int | None:
    """Majority severity class among scar pixels inside the patch window."""
    sev_win = extract_window(severity, row, col, size)
    scar_win = extract_window(scar, row, col, size)
    valid = (scar_win > 0) & (sev_win > 0)
    if valid.sum() == 0:
        return None
    values, counts = np.unique(sev_win[valid], return_counts=True)
    return int(values[counts.argmax()])


def invalid_lulc_fraction(lulc_win: np.ndarray, class_ids: list[int]) -> float:
    valid_set = set(class_ids)
    flat = lulc_win.ravel()
    invalid = sum(int(v) not in valid_set or v <= 0 for v in flat)
    return float(invalid) / max(len(flat), 1)


def patch_label(
    severity: np.ndarray,
    scar: np.ndarray,
    row: int,
    col: int,
    size: int,
    strategy: str,
) -> tuple[int | None, int | None]:
    """
    Return (center_severity, binary_label) or (None, None) if invalid.

    Primary strategy uses center pixel; optional dominant uses scar pixels only.
    """
    center_sev = int(severity[row, col])
    if strategy == "dominant":
        center_sev = dominant_severity_in_patch(severity, scar, row, col, size) or 0
    if center_sev <= 0:
        return None, None
    binary = severity_to_binary_label(center_sev)
    if binary is None:
        return None, None
    return center_sev, binary


def iter_stride_centers(h: int, w: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(0, h, stride)
    cols = np.arange(0, w, stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return rr.ravel(), cc.ravel()


def extract_cnn_patches_for_fire(
    fire_id: str,
    lulc: np.ndarray,
    severity: np.ndarray,
    scar: np.ndarray,
    class_ids: list[int],
    combustible_ids: set[int],
    severity_class_map: dict[int, int],
    criteria: CNNPatchCriteria,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract CNN patches with configurable stride.

    Returns X (N,C,H,W), y (N,) binary labels, metadata DataFrame.
    """
    severity_mapped = remap_severity(severity, severity_class_map)
    lulc_onehot_full = lulc_to_onehot(lulc, class_ids)
    patch_criteria = PatchCriteria(
        size=criteria.size,
        min_burn_fraction=criteria.min_burn_fraction,
        min_valid_severity_fraction=criteria.min_valid_severity_fraction,
        max_outside_scar_fraction=criteria.max_outside_scar_fraction,
        random_seed=criteria.random_seed,
        severity_classes=(1, 2),
    )

    h, w = scar.shape
    rows, cols = iter_stride_centers(h, w, criteria.stride)

    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict] = []

    for row, col in zip(rows, cols, strict=False):
        row_i, col_i = int(row), int(col)
        sample = evaluate_center(scar, severity_mapped, row_i, col_i, patch_criteria)
        if sample is None:
            continue

        lulc_win = extract_window(lulc, row_i, col_i, criteria.size)
        if invalid_lulc_fraction(lulc_win, class_ids) > criteria.max_invalid_lulc_fraction:
            continue

        center_sev, binary = patch_label(
            severity_mapped,
            scar,
            row_i,
            col_i,
            criteria.size,
            criteria.label_strategy,
        )
        if binary is None:
            continue

        props = class_proportions(lulc_win, class_ids)
        dominant_lulc = max(props, key=props.get)
        metrics = summarize_patch(lulc_win, class_ids, combustible_ids)

        x_list.append(extract_window(lulc_onehot_full, row_i, col_i, criteria.size))
        y_list.append(binary)
        meta_rows.append(
            {
                "scar_id": fire_id,
                "fire_id": fire_id,
                "row": row_i,
                "col": col_i,
                "center_severity": center_sev,
                "binary_label": binary,
                "burn_fraction": sample.burn_fraction,
                "valid_severity_fraction": sample.valid_severity_fraction,
                "dominant_lulc": dominant_lulc,
                "richness": metrics["richness"],
                "combustible_fraction": metrics["combustible_fraction"],
                "edge_density": metrics["edge_density"],
                **{f"prop_{cid}": props[cid] for cid in class_ids},
            }
        )

    if not x_list:
        empty_x = np.zeros((0, len(class_ids), criteria.size, criteria.size), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        return empty_x, empty_y, pd.DataFrame()

    x = np.stack(x_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)
    return x, y, meta


def save_cnn_patch_bundle(
    output_path: Path,
    fire_id: str,
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fire_id=fire_id,
        x=x,
        y=y,
        meta=meta.to_dict(orient="list"),
    )


def load_cnn_patch_bundle(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    bundle = {k: data[k] for k in data.files}
    if "meta" in bundle and isinstance(bundle["meta"], np.ndarray):
        bundle["meta"] = pd.DataFrame(bundle["meta"].item())
    return bundle
