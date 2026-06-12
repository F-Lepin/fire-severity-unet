"""Ingest multi-band fire scar GeoTIFFs into the pipeline layout."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio

from fire_severity.data.encoding import remap_lulc
from fire_severity.data.severity import apply_severity_class_map, classify_dnbr

FIRE_ID_RE = re.compile(r"fire_top45_(scar_\d+)_", re.IGNORECASE)


def parse_fire_id(path: Path) -> str:
    match = FIRE_ID_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse fire id from filename: {path.name}")
    return match.group(1).lower()


def _band_index(src: rasterio.DatasetReader, names: list[str], fallback: int) -> int:
    descriptions = [d.lower() if d else "" for d in (src.descriptions or [])]
    for name in names:
        name_l = name.lower()
        for i, desc in enumerate(descriptions, start=1):
            if name_l in desc:
                return i
    return fallback


def ingest_multiband_raster(
    source_path: Path,
    out_dir: Path,
    *,
    band_lulc: int | None = None,
    band_dnbr: int | None = None,
    band_scar: int | None = None,
    lulc_remap: dict[int, int] | None = None,
    severity_mode: str = "dnbr_thresholds",
    class_ranges: dict[int, tuple[float, float]] | None = None,
    dnbr_value_scale: float = 1.0,
    severity_class_map: dict[int, int] | None = None,
    dnbr_nodata: int | float | None = None,
) -> dict:
    """
    Read a 3-band stack and write ``lulc.tif``, ``severity.tif``, ``scar.tif``.

    Returns summary stats for logging / QA.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_path) as src:
        idx_lulc = band_lulc or _band_index(src, ["lulc"], 1)
        idx_dnbr = band_dnbr or _band_index(src, ["dnbr"], 2)
        idx_scar = band_scar or _band_index(src, ["scar", "mask"], 3)

        lulc = src.read(idx_lulc)
        dnbr = src.read(idx_dnbr)
        scar = src.read(idx_scar)
        profile = src.profile.copy()
        profile.update(count=1, compress="lzw")

    scar_bin = (scar > 0).astype(np.int16)

    if lulc_remap:
        lulc_out = remap_lulc(lulc, {int(k): int(v) for k, v in lulc_remap.items()})
    else:
        lulc_out = lulc.astype(np.int16)

    if severity_mode == "dnbr_thresholds":
        if not class_ranges:
            raise ValueError("class_ranges required when severity_mode=dnbr_thresholds")
        severity = classify_dnbr(
            dnbr,
            scar_bin,
            class_ranges,
            nodata=dnbr_nodata,
            value_scale=dnbr_value_scale,
        )
    elif severity_mode == "class_map":
        if not severity_class_map:
            raise ValueError("severity_class_map required when severity_mode=class_map")
        severity = apply_severity_class_map(
            dnbr,
            scar_bin,
            {int(k): int(v) for k, v in severity_class_map.items()},
        )
    else:
        raise ValueError(f"Unknown severity_mode: {severity_mode}")

    inside = scar_bin > 0
    summary = {
        "lulc_unique": sorted(int(v) for v in np.unique(lulc_out)),
        "dnbr_unique_in_scar": sorted(int(v) for v in np.unique(dnbr[inside])) if inside.any() else [],
        "severity_unique_in_scar": sorted(int(v) for v in np.unique(severity[inside])) if inside.any() else [],
        "n_scar_pixels": int(inside.sum()),
    }

    for name, arr in [("lulc.tif", lulc_out), ("severity.tif", severity), ("scar.tif", scar_bin)]:
        out_profile = {**profile, "dtype": arr.dtype}
        with rasterio.open(out_dir / name, "w", **out_profile) as dst:
            dst.write(arr, 1)

    return summary
