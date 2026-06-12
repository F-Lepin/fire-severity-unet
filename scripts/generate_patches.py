#!/usr/bin/env python3
"""Generate 32×32 patches for one or all fires."""

from __future__ import annotations

import argparse
from pathlib import Path

from fire_severity.config import lulc_class_ids, load_config, severity_class_ids
from fire_severity.data.alignment import load_aligned_stack
from fire_severity.data.patches import (
    PatchCriteria,
    build_patch_arrays,
    save_patch_bundle,
    select_patches,
)


def process_fire(fire_id: str, cfg: dict) -> None:
    proc_dir = Path(cfg["data"]["processed_root"]) / fire_id
    lulc, severity, scar, _, _ = load_aligned_stack(
        proc_dir / "lulc.tif",
        proc_dir / "severity.tif",
        proc_dir / "scar.tif",
    )

    pcfg = cfg["patches"]
    criteria = PatchCriteria(
        size=pcfg["size"],
        min_burn_fraction=pcfg["min_burn_fraction"],
        min_valid_severity_fraction=pcfg["min_valid_severity_fraction"],
        max_outside_scar_fraction=pcfg["max_outside_scar_fraction"],
        samples_per_fire=pcfg["samples_per_fire"],
        balance_severity=pcfg["balance_severity"],
        random_seed=pcfg["random_seed"],
        severity_classes=tuple(severity_class_ids(cfg)),
    )

    class_ids = lulc_class_ids(cfg)
    severity_map = {int(k): int(v) for k, v in cfg["severity"]["class_map"].items()}

    samples = select_patches(fire_id, scar, severity, criteria)
    if not samples:
        print(f"[{fire_id}] No patches passed selection criteria.")
        return

    x, y, loss_mask, _ = build_patch_arrays(
        lulc, severity, scar, samples, class_ids, severity_map, pcfg["size"]
    )

    out_path = Path(cfg["data"]["patches_root"]) / f"{fire_id}_patches.npz"
    save_patch_bundle(out_path, fire_id, x, y, loss_mask, samples)
    print(f"[{fire_id}] Saved {len(samples)} patches → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training patches.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fire-id", default=None, help="Process one fire; default = all processed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.fire_id:
        process_fire(args.fire_id, cfg)
        return

    proc_root = Path(cfg["data"]["processed_root"])
    for fire_dir in sorted(p for p in proc_root.iterdir() if p.is_dir()):
        process_fire(fire_dir.name, cfg)


if __name__ == "__main__":
    main()
