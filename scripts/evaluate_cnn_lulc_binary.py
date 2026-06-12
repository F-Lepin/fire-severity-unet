#!/usr/bin/env python3
"""Evaluate a trained CNN LULC patch classifier on held-out fires."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fire_severity.config import load_config
from fire_severity.cnn.dataset import LULCPatchDataset, collate_cnn_batch
from fire_severity.cnn.evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from fire_severity.cnn.model import SmallPatchCNN
from fire_severity.cnn.train import predict_loader
from fire_severity.training.trainer import resolve_device
from fire_severity.validation.splits import holdout_fire_splits, kfold_fire_splits, leave_one_fire_out_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN LULC classifier.")
    parser.add_argument("--config", default="config/cnn_lulc_binary.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["validation"]
    run_name = vcfg.get("run_name", f"fold_{args.fold}")
    tcfg = cfg["training"]
    out_dir = Path(tcfg.get("outputs_dir", "outputs_cnn_lulc_binary")) / run_name
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "model_best.pt"

    patch_files = sorted(Path(cfg["data"]["patches_root"]).glob("*_cnn_patches.npz"))
    strategy = vcfg["strategy"]
    if strategy == "leave_one_fire_out":
        _, val_files, _ = leave_one_fire_out_splits(patch_files)[args.fold]
    elif strategy == "holdout_fires":
        _, val_files, _, _ = holdout_fire_splits(
            patch_files,
            val_fraction=float(vcfg.get("val_fraction", 0.30)),
            seed=int(vcfg.get("random_seed", cfg["patches"]["random_seed"])),
        )[args.fold]
    else:
        _, val_files, _ = kfold_fire_splits(
            patch_files, vcfg.get("n_folds", 5), cfg["patches"]["random_seed"]
        )[args.fold]

    device = resolve_device(tcfg["device"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SmallPatchCNN(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    ds = LULCPatchDataset(val_files)
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate_cnn_batch)
    probs, preds, y_true = predict_loader(model, loader, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_predictions(y_true, preds, probs).to_csv(out_dir / "metrics_test.csv", index=False)
    plot_confusion_matrix(y_true, preds, out_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, probs, out_dir / "roc_curve.png")
    plot_precision_recall_curve(y_true, probs, out_dir / "precision_recall_curve.png")
    print(f"Metrics saved to {out_dir}")


if __name__ == "__main__":
    main()
