#!/usr/bin/env python3
"""Train U-Net with fire-level validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fire_severity.config import load_config
from fire_severity.interpretability.analysis import plot_training_history
from fire_severity.training.trainer import TrainConfig, train_model
from fire_severity.validation.splits import (
    holdout_fire_splits,
    kfold_fire_splits,
    leave_one_fire_out_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train severity U-Net.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for CV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patch_files = sorted(Path(cfg["data"]["patches_root"]).glob("*_patches.npz"))
    if not patch_files:
        raise FileNotFoundError("No patch files found. Run generate_patches.py first.")

    vcfg = cfg["validation"]
    strategy = vcfg["strategy"]
    split_meta: dict = {"strategy": strategy}

    if strategy == "leave_one_fire_out":
        splits = leave_one_fire_out_splits(patch_files)
        if args.fold >= len(splits):
            raise ValueError(f"fold {args.fold} out of range (n_fires={len(splits)})")
        train_files, val_files, held_out = splits[args.fold]
        split_meta["held_out_fire"] = held_out
        split_meta["train_fires"] = "all except held_out"
        split_meta["val_fires"] = [held_out]
        print(f"Validation: leave-out fire {held_out}")
    elif strategy == "holdout_fires":
        splits = holdout_fire_splits(
            patch_files,
            val_fraction=float(vcfg.get("val_fraction", 0.30)),
            seed=int(vcfg.get("random_seed", cfg["patches"]["random_seed"])),
        )
        if args.fold >= len(splits):
            raise ValueError(f"fold {args.fold} out of range (holdout has 1 split)")
        train_files, val_files, train_fires, val_fires = splits[args.fold]
        split_meta["val_fraction"] = vcfg.get("val_fraction", 0.30)
        split_meta["train_fires"] = train_fires
        split_meta["val_fires"] = val_fires
        print(f"Validation: holdout {len(val_fires)}/{len(train_fires) + len(val_fires)} fires")
        print(f"  train: {train_fires}")
        print(f"  val:   {val_fires}")
    else:
        splits = kfold_fire_splits(
            patch_files, vcfg.get("n_folds", 5), cfg["patches"]["random_seed"]
        )
        if args.fold >= len(splits):
            raise ValueError(f"fold {args.fold} out of range (n_folds={len(splits)})")
        train_files, val_files, val_fires = splits[args.fold]
        split_meta["val_fires"] = val_fires
        print(f"Validation fires: {val_fires}")

    split_meta["n_train_patch_files"] = len(train_files)
    split_meta["n_val_patch_files"] = len(val_files)
    print(f"  patches: train={len(train_files)} val={len(val_files)}")

    tcfg = cfg["training"]
    run_name = vcfg.get("run_name", f"fold_{args.fold}")
    train_cfg = TrainConfig(
        batch_size=tcfg["batch_size"],
        epochs=tcfg["epochs"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        num_workers=tcfg["num_workers"],
        device=tcfg["device"],
        checkpoint_dir=Path(tcfg["checkpoint_dir"]) / run_name,
    )

    model, history = train_model(train_files, val_files, cfg["model"], train_cfg)

    out_dir = Path(tcfg.get("outputs_dir", "outputs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, out_dir / "training_curves.png")
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split_meta,
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "val_acc": history.val_acc,
            },
            f,
            indent=2,
        )
    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
