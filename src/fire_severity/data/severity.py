"""dNBR and severity raster classification."""

from __future__ import annotations

import numpy as np


def classify_dnbr(
    dnbr: np.ndarray,
    scar: np.ndarray,
    class_ranges: dict[int, tuple[float, float]],
    nodata: int | float | None = None,
    value_scale: float = 1.0,
) -> np.ndarray:
    """
    Reclassify continuous dNBR into severity classes inside the burn scar.

    ``class_ranges``: inclusive (min, max) per class id, e.g.
      {1: (0.100, 0.269), 2: (0.270, 0.439), ...}

    Pixels inside the scar below the minimum of class 1, or with no matching
    range, remain 0 (ignored in training loss).
    """
    if not class_ranges:
        raise ValueError("class_ranges must define at least one severity class.")

    severity = np.zeros(dnbr.shape, dtype=np.int16)
    inside = scar > 0
    if nodata is not None:
        inside &= dnbr != nodata

    vals = dnbr[inside].astype(np.float64) * value_scale
    if vals.size == 0:
        return severity

    cls = np.zeros(vals.shape, dtype=np.int16)
    for class_id in sorted(int(k) for k in class_ranges):
        lo, hi = class_ranges[class_id]
        cls[(vals >= lo) & (vals <= hi)] = class_id

    severity[inside] = cls
    return severity


def apply_severity_class_map(
    values: np.ndarray,
    scar: np.ndarray,
    class_map: dict[int, int],
    nodata_values: set[int] | None = None,
) -> np.ndarray:
    """Map existing integer codes to model classes."""
    from fire_severity.data.encoding import remap_severity

    mapped = remap_severity(values.astype(np.int64), class_map, nodata_values)
    mapped[scar <= 0] = 0
    return mapped.astype(np.int16)
