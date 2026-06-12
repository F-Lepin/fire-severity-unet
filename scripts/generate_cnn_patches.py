#!/usr/bin/env python3
"""Generate CNN LULC patches (stride grid, center-pixel binary labels)."""

from __future__ import annotations

import argparse
from pathlib import Path

from fire_severity.config import combustible_ids, lulc_class_ids, load_config
from fire_severity.cnn.patches import CNNPatchCriteria, extract_cnn_patches_for_fire, save_cnn_patch_bundle
from fire_severity.data.alignment import load_aligned_stack


def process_fire(fire_id: str, cfg: dict) -> None:
    proc_dir = Path(cfg["data"]["processed_root"]) / fire_id
    lulc, severity, scar, _, _ = load_aligned_stack(
        proc_dir / "lulc.tif",
        proc_dir / "severity.tif",
        proc_dir / "scar.tif",
    )

    pcfg = cfg["patches"]
    criteria = CNNPatchCriteria(
        size=pcfg["size"],
        stride=pcfg.get("stride", 16),
        min_burn_fraction=pcfg["min_burn_fraction"],
        min_valid_severity_fraction=pcfg["min_valid_severity_fraction"],
        max_outside_scar_fraction=pcfg["max_outside_scar_fraction"],
        max_invalid_lulc_fraction=pcfg.get("max_invalid_lulc_fraction", 0.10),
        label_strategy=pcfg.get("label_strategy", "center"),
        random_seed=pcfg.get("random_seed", 42),
    )

    class_ids = lulc_class_ids(cfg)
    comb_ids = combustible_ids(cfg)
    severity_map = {int(k): int(v) for k, v in cfg["severity"]["class_map"].items()}

    x, y, meta = extract_cnn_patches_for_fire(
        fire_id,
        lulc,
        severity,
        scar,
        class_ids,
        comb_ids,
        severity_map,
        criteria,
    )
    if len(y) == 0:
        print(f"[{fire_id}] No patches passed selection criteria.")
        return

    out_path = Path(cfg["data"]["patches_root"]) / f"{fire_id}_cnn_patches.npz"
    save_cnn_patch_bundle(out_path, fire_id, x, y, meta)
    print(f"[{fire_id}] Saved {len(y)} patches → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CNN LULC patch bundles.")
    parser.add_argument("--config", default="config/cnn_lulc_binary.yaml")
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
