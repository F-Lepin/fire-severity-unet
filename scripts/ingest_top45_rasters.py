#!/usr/bin/env python3
"""Split multi-band fire_top45 GeoTIFFs into lulc / severity / scar rasters."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fire_severity.config import load_config, severity_class_ranges
from fire_severity.data.ingest import ingest_multiband_raster, parse_fire_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest fire_top45 multi-band rasters.")
    parser.add_argument("--config", default="config/leftraru.yaml")
    parser.add_argument("--fire-id", default=None, help="Process one fire, e.g. scar_01")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    icfg = cfg.get("ingest", {})
    pattern = Path(icfg["source_glob"])
    if pattern.is_absolute():
        sources = sorted(pattern.parent.glob(pattern.name))
    else:
        sources = sorted(Path(".").glob(str(pattern)))

    if args.fire_id:
        sources = [p for p in sources if parse_fire_id(p) == args.fire_id.lower()]
        if not sources:
            raise FileNotFoundError(f"No source raster for fire-id {args.fire_id}")

    if not sources:
        raise FileNotFoundError(f"No rasters matched: {pattern}")

    lulc_remap = {int(k): int(v) for k, v in cfg.get("lulc", {}).get("remap", {}).items()}
    sev_cfg = cfg.get("severity", {})
    severity_mode = sev_cfg.get("mode", "dnbr_thresholds")
    class_ranges = severity_class_ranges(cfg) if "class_ranges" in sev_cfg else None
    dnbr_value_scale = float(sev_cfg.get("dnbr_value_scale", 1.0))
    severity_class_map = {int(k): int(v) for k, v in sev_cfg.get("class_map", {}).items()}
    dnbr_nodata = sev_cfg.get("dnbr_nodata")

    fires_root = Path(cfg["data"]["fires_root"])
    processed_root = Path(cfg["data"]["processed_root"])
    copy_to_processed = icfg.get("copy_to_processed", True)

    for src_path in sources:
        fire_id = parse_fire_id(src_path)
        out_dir = fires_root / fire_id
        print(f"\n[{fire_id}] {src_path.name}")

        if args.dry_run:
            print(f"  → would write {out_dir}/")
            continue

        summary = ingest_multiband_raster(
            src_path,
            out_dir,
            band_lulc=icfg.get("band_lulc"),
            band_dnbr=icfg.get("band_dnbr"),
            band_scar=icfg.get("band_scar"),
            lulc_remap=lulc_remap or None,
            severity_mode=severity_mode,
            class_ranges=class_ranges,
            dnbr_value_scale=dnbr_value_scale,
            severity_class_map=severity_class_map,
            dnbr_nodata=dnbr_nodata,
        )
        print(f"  LULC classes : {summary['lulc_unique']}")
        print(f"  dNBR in scar : {summary['dnbr_unique_in_scar']}")
        print(f"  severity     : {summary['severity_unique_in_scar']} ({summary['n_scar_pixels']} px)")

        if len(summary["severity_unique_in_scar"]) < 2:
            print("  ⚠ Solo una clase de severidad dentro de la cicatriz — revisar banda dNBR.")

        if copy_to_processed:
            proc_dir = processed_root / fire_id
            proc_dir.mkdir(parents=True, exist_ok=True)
            for name in ("lulc.tif", "severity.tif", "scar.tif"):
                shutil.copy2(out_dir / name, proc_dir / name)
            print(f"  → copied to {proc_dir}")


if __name__ == "__main__":
    main()
