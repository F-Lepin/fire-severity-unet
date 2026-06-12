"""Raster alignment: reproject, resample and clip to a common grid."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
from rasterio.transform import Affine


def _open_array(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        meta["transform"] = src.transform
        meta["crs"] = src.crs
        meta["nodata"] = src.nodata
    return data, meta


def align_rasters(
    paths: Sequence[Path],
    output_dir: Path,
    target_crs: str,
    target_resolution: float,
    resampling: Resampling = Resampling.nearest,
) -> list[Path]:
    """
    Reproject and resample multiple rasters to a shared grid.

    Uses the union bounds of all inputs. Categorical layers (LULC, severity, scar)
    should use nearest-neighbor resampling.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []
    metas: list[dict] = []

    for path in paths:
        arr, meta = _open_array(path)
        arrays.append(arr)
        metas.append(meta)

    bounds = None
    for meta in metas:
        left, bottom, right, top = rasterio.transform.array_bounds(
            meta["height"], meta["width"], meta["transform"]
        )
        if bounds is None:
            bounds = [left, bottom, right, top]
        else:
            bounds = [
                min(bounds[0], left),
                min(bounds[1], bottom),
                max(bounds[2], right),
                max(bounds[3], top),
            ]

    assert bounds is not None
    dst_crs = rasterio.crs.CRS.from_string(target_crs)
    transform, width, height = calculate_default_transform(
        metas[0]["crs"],
        dst_crs,
        metas[0]["width"],
        metas[0]["height"],
        *bounds,
        resolution=target_resolution,
    )

    out_paths: list[Path] = []
    profile = {
        "driver": "GTiff",
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
        "count": 1,
        "compress": "lzw",
    }

    for path, src_arr, src_meta in zip(paths, arrays, metas):
        dst = np.zeros((height, width), dtype=src_arr.dtype)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src_meta["transform"],
            src_crs=src_meta["crs"],
            src_nodata=src_meta.get("nodata"),
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=src_meta.get("nodata"),
            resampling=resampling,
        )
        out_path = output_dir / path.name
        dtype = src_arr.dtype
        out_profile = {**profile, "dtype": dtype, "nodata": src_meta.get("nodata")}
        with rasterio.open(out_path, "w", **out_profile) as dst_src:
            dst_src.write(dst, 1)
        out_paths.append(out_path)

    return out_paths


def load_aligned_stack(
    lulc_path: Path,
    severity_path: Path,
    scar_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Affine, str | None]:
    """Load three aligned single-band rasters."""
    with rasterio.open(lulc_path) as lulc_src:
        lulc = lulc_src.read(1)
        transform = lulc_src.transform
        crs = lulc_src.crs.to_string() if lulc_src.crs else None
    with rasterio.open(severity_path) as sev_src:
        severity = sev_src.read(1)
    with rasterio.open(scar_path) as scar_src:
        scar = scar_src.read(1)
    if lulc.shape != severity.shape or lulc.shape != scar.shape:
        raise ValueError("Rasters must share shape after alignment.")
    return lulc, severity, scar, transform, crs
