#!/usr/bin/env python3
"""Train U-Net with fire-level validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fire_severity.config import load_config
from fire_severity.interpretability.analysis import plot_training_history
from fire_severity.training.trainer import TrainConfig, train_model
from fire_severity.validation.splits import kfold_fire_splits, leave_one_fire_out_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train severity U-Net.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fold", type=int, default=0, help="Fold index for CV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    patch_files = sorted(Path(cfg["data"]["patches_root"]).glob("*_patches.npz"))
    if not patch_files:
        raise FileNotFoundError("No patch files found. Run generate_patches.py first.")

    strategy = cfg["validation"]["strategy"]
    if strategy == "leave_one_fire_out":
        splits = leave_one_fire_out_splits(patch_files)
        if args.fold >= len(splits):
            raise ValueError(f"fold {args.fold} out of range (n_fires={len(splits)})")
        train_files, val_files, held_out = splits[args.fold]
        print(f"Validation: leave-out fire {held_out}")
    else:
        splits = kfold_fire_splits(patch_files, cfg["validation"]["n_folds"], cfg["patches"]["random_seed"])
        if args.fold >= len(splits):
            raise ValueError(f"fold {args.fold} out of range (n_folds={len(splits)})")
        train_files, val_files, val_fires = splits[args.fold]
        print(f"Validation fires: {val_fires}")

    tcfg = cfg["training"]
    train_cfg = TrainConfig(
        batch_size=tcfg["batch_size"],
        epochs=tcfg["epochs"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        num_workers=tcfg["num_workers"],
        device=tcfg["device"],
        checkpoint_dir=Path(tcfg["checkpoint_dir"]) / f"fold_{args.fold}",
    )

    model, history = train_model(train_files, val_files, cfg["model"], train_cfg)

    out_dir = Path(tcfg.get("outputs_dir", "outputs")) / f"fold_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, out_dir / "training_curves.png")
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
                "val_acc": history.val_acc,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
