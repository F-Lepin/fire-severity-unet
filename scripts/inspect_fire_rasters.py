#!/usr/bin/env python3
"""QA: band metadata and unique values for fire_top45 rasters."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

from fire_severity.config import load_config
from fire_severity.data.ingest import parse_fire_id


def _finite_class_codes(arr: np.ndarray) -> list[int]:
    """Unique integer class codes, ignoring NaN/inf (float rasters)."""
    flat = arr[np.isfinite(arr)].ravel()
    if flat.size == 0:
        return []
    return sorted({int(round(v)) for v in np.unique(flat)})


def _scar_inside(scar: np.ndarray) -> np.ndarray:
    return np.isfinite(scar) & (scar > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect source fire rasters.")
    parser.add_argument("--config", default="config/leftraru.yaml")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pattern = Path(cfg["ingest"]["source_glob"])
    if pattern.is_absolute():
        sources = sorted(pattern.parent.glob(pattern.name))
    else:
        sources = sorted(Path(".").glob(str(pattern)))

    if args.max_files:
        sources = sources[: args.max_files]

    if not sources:
        raise FileNotFoundError(f"No files for pattern {pattern}")

    lulc_counter: Counter[int] = Counter()
    scar_counter: Counter[int] = Counter()
    dnbr_samples: list[np.ndarray] = []

    print(f"Found {len(sources)} raster(s)\n")

    with rasterio.open(sources[0]) as s0:
        print("Example file:", sources[0].name)
        print("  shape:", s0.height, "x", s0.width, " crs:", s0.crs, " res:", s0.res)
        for i in range(1, s0.count + 1):
            d = s0.descriptions[i - 1] if s0.descriptions else None
            arr = s0.read(i).astype(np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                print(
                    f"  band {i} ({d!r}): min={finite.min():.4f} max={finite.max():.4f} "
                    f"dtype={s0.dtypes[i - 1]} nodata={s0.nodatavals[i - 1]}"
                )
            else:
                print(f"  band {i} ({d!r}): all nodata dtype={s0.dtypes[i - 1]}")

    print("\nAggregating across files...")
    for path in sources:
        fire_id = parse_fire_id(path)
        with rasterio.open(path) as src:
            lulc = src.read(1)
            dnbr = src.read(2)
            scar = src.read(3)

        inside = _scar_inside(scar)
        for code in _finite_class_codes(lulc):
            lulc_counter[code] += 1
        for code in _finite_class_codes(scar):
            scar_counter[code] += 1

        if inside.any():
            v = dnbr[inside].astype(np.float64)
            v = v[np.isfinite(v)]
            dnbr_samples.append(v)
            print(
                f"  {fire_id}: scar_px={inside.sum()} "
                f"dnbr=[{v.min():.3f}, {v.max():.3f}] "
                f"p50={np.percentile(v, 50):.3f}"
            )
        else:
            print(f"  {fire_id}: no valid scar pixels")

    print("\n--- LULC codes MapBiomas (present in ≥1 file) ---")
    for code, n in sorted(lulc_counter.items()):
        print(f"  {code}: {n} file(s)")

    print("\n--- scar_mask values (finite) ---")
    print(" ", sorted(scar_counter.keys()))

    if dnbr_samples:
        all_dnbr = np.concatenate(dnbr_samples)
        print("\n--- dNBR inside scar (pooled) ---")
        print(f"  range: [{all_dnbr.min():.3f}, {all_dnbr.max():.3f}]")
        print(f"  percentiles: {np.percentile(all_dnbr, [5, 25, 50, 75, 95]).round(3)}")


if __name__ == "__main__":
    main()
