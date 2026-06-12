#!/usr/bin/env python3
"""Align LULC, severity and burn-scar rasters per fire."""

from __future__ import annotations

import argparse
from pathlib import Path

from rasterio.enums import Resampling

from fire_severity.config import load_config
from fire_severity.data.alignment import align_rasters


def main() -> None:
    parser = argparse.ArgumentParser(description="Align fire rasters to a common grid.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fire-id", required=True, help="Fire folder name under data/raw/fires/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fire_dir = Path(cfg["data"]["fires_root"]) / args.fire_id
    out_dir = Path(cfg["data"]["processed_root"]) / args.fire_id
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        fire_dir / "lulc.tif",
        fire_dir / "severity.tif",
        fire_dir / "scar.tif",
    ]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Missing raster: {p}")

    align_rasters(
        inputs,
        out_dir,
        target_crs=cfg["data"]["target_crs"],
        target_resolution=cfg["data"]["target_resolution"],
        resampling=Resampling.nearest,
    )
    print(f"Aligned rasters written to {out_dir}")


if __name__ == "__main__":
    main()
