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
    dnbr_in_scar: Counter[int] = Counter()
    scar_counter: Counter[int] = Counter()

    print(f"Found {len(sources)} raster(s)\n")

    with rasterio.open(sources[0]) as s0:
        print("Example file:", sources[0].name)
        print("  shape:", s0.height, "x", s0.width, " crs:", s0.crs, " res:", s0.res)
        for i in range(1, s0.count + 1):
            d = s0.descriptions[i - 1] if s0.descriptions else None
            arr = s0.read(i)
            print(f"  band {i} ({d!r}): min={arr.min()} max={arr.max()} dtype={arr.dtype}")

    print("\nAggregating unique values across files...")
    for path in sources:
        fire_id = parse_fire_id(path)
        with rasterio.open(path) as src:
            lulc = src.read(1)
            dnbr = src.read(2)
            scar = src.read(3)
        inside = scar > 0
        lulc_counter.update(int(v) for v in np.unique(lulc))
        scar_counter.update(int(v) for v in np.unique(scar))
        if inside.any():
            dnbr_in_scar.update(int(v) for v in np.unique(dnbr[inside]))
        print(f"  {fire_id}: scar_px={inside.sum()} dnbr_in_scar={sorted(np.unique(dnbr[inside]))}")

    print("\n--- LULC codes (all files) ---")
    for code, n in sorted(lulc_counter.items()):
        print(f"  {code}: {n} file(s)")

    print("\n--- dNBR inside scar (union) ---")
    print(" ", sorted(dnbr_in_scar.keys()))

    print("\n--- scar_mask values ---")
    print(" ", sorted(scar_counter.keys()))


if __name__ == "__main__":
    main()
