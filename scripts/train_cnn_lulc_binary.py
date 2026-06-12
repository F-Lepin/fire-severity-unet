#!/usr/bin/env python3
"""
Train CNN LULC patch classifier (binary severity at patch center).

MVP flow:
  1. Optionally generate patches (--generate-patches)
  2. Fire-level train/val split (no spatial leakage)
  3. Train small CNN on LULC one-hot
  4. Evaluate on held-out fires
  5. Export interpretability outputs
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from torch.utils.data import DataLoader

from fire_severity.config import load_config, num_lulc_channels
from fire_severity.cnn.dataset import LULCPatchDataset, collate_cnn_batch
from fire_severity.cnn.evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_log,
)
from fire_severity.cnn.interpret import run_interpretability
from fire_severity.cnn.model import SmallPatchCNN
from fire_severity.cnn.train import CNNTrainConfig, predict_loader, train_cnn_model
from fire_severity.training.trainer import resolve_device
from fire_severity.validation.splits import holdout_fire_splits, kfold_fire_splits, leave_one_fire_out_splits


def resolve_split(cfg: dict, patch_files: list[Path], fold: int) -> tuple[list[Path], list[Path], dict]:
    vcfg = cfg["validation"]
    strategy = vcfg["strategy"]
    split_meta: dict = {"strategy": strategy}

    if strategy == "leave_one_fire_out":
        splits = leave_one_fire_out_splits(patch_files)
        if fold >= len(splits):
            raise ValueError(f"fold {fold} out of range (n_fires={len(splits)})")
        train_files, val_files, held_out = splits[fold]
        split_meta["held_out_fire"] = held_out
        split_meta["val_fires"] = [held_out]
    elif strategy == "holdout_fires":
        splits = holdout_fire_splits(
            patch_files,
            val_fraction=float(vcfg.get("val_fraction", 0.30)),
            seed=int(vcfg.get("random_seed", cfg["patches"]["random_seed"])),
        )
        if fold >= len(splits):
            raise ValueError(f"fold {fold} out of range (holdout has 1 split)")
        train_files, val_files, train_fires, val_fires = splits[fold]
        split_meta["val_fraction"] = vcfg.get("val_fraction", 0.30)
        split_meta["train_fires"] = train_fires
        split_meta["val_fires"] = val_fires
    else:
        splits = kfold_fire_splits(
            patch_files, vcfg.get("n_folds", 5), cfg["patches"]["random_seed"]
        )
        if fold >= len(splits):
            raise ValueError(f"fold {fold} out of range (n_folds={len(splits)})")
        train_files, val_files, val_fires = splits[fold]
        split_meta["val_fires"] = val_fires
        split_meta["train_fires"] = "see patch files"

    split_meta["n_train_patch_files"] = len(train_files)
    split_meta["n_val_patch_files"] = len(val_files)
    return train_files, val_files, split_meta


def maybe_generate_patches(config_path: str, fire_id: str | None) -> None:
    cmd = [sys.executable, "scripts/generate_cnn_patches.py", "--config", config_path]
    if fire_id:
        cmd.extend(["--fire-id", fire_id])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN LULC binary patch classifier.")
    parser.add_argument("--config", default="config/cnn_lulc_binary.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--generate-patches", action="store_true", help="Run patch extraction first")
    parser.add_argument("--fire-id", default=None, help="Only process one fire when generating patches")
    parser.add_argument("--skip-interpret", action="store_true")
    args = parser.parse_args()

    if args.generate_patches:
        maybe_generate_patches(args.config, args.fire_id)

    cfg = load_config(args.config)
    patch_files = sorted(Path(cfg["data"]["patches_root"]).glob("*_cnn_patches.npz"))
    if not patch_files:
        raise FileNotFoundError(
            "No CNN patch files found. Run with --generate-patches or execute scripts/generate_cnn_patches.py."
        )

    train_files, val_files, split_meta = resolve_split(cfg, patch_files, args.fold)
    print(f"Train patch files: {len(train_files)} | Val/test patch files: {len(val_files)}")

    vcfg = cfg["validation"]
    run_name = vcfg.get("run_name", f"fold_{args.fold}")
    tcfg = cfg["training"]
    ccfg = cfg.get("cnn", {})

    model_cfg = {
        "in_channels": num_lulc_channels(cfg),
        "num_classes": ccfg.get("num_classes", 2),
        "dropout": ccfg.get("dropout", 0.3),
    }
    train_cfg = CNNTrainConfig(
        batch_size=tcfg["batch_size"],
        epochs=tcfg["epochs"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        num_workers=tcfg["num_workers"],
        device=tcfg["device"],
        checkpoint_dir=Path(tcfg["checkpoint_dir"]) / run_name,
        use_class_weights=ccfg.get("use_class_weights", True),
        use_weighted_sampler=ccfg.get("use_weighted_sampler", False),
    )

    model, history, log_df = train_cnn_model(train_files, val_files, model_cfg, train_cfg)

    out_dir = Path(tcfg.get("outputs_dir", "outputs_cnn_lulc_binary")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_cfg.checkpoint_dir / "model_best.pt", out_dir / "model_best.pt")
    log_df.to_csv(out_dir / "training_log.csv", index=False)
    plot_training_log(log_df, out_dir / "training_curves.png")

    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    device = resolve_device(tcfg["device"])
    val_ds = LULCPatchDataset(val_files)
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        collate_fn=collate_cnn_batch,
    )
    probs, preds, y_true = predict_loader(model, val_loader, device)

    metrics_df = evaluate_predictions(y_true, preds, probs)
    metrics_df.to_csv(out_dir / "metrics_test.csv", index=False)
    plot_confusion_matrix(y_true, preds, out_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, probs, out_dir / "roc_curve.png")
    plot_precision_recall_curve(y_true, probs, out_dir / "precision_recall_curve.png")

    if not args.skip_interpret:
        run_interpretability(model, val_files, cfg, out_dir, device)

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
