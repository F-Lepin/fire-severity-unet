#!/usr/bin/env python3
"""Post-training interpretability and congress-ready figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fire_severity.config import combustible_ids, lulc_class_ids, load_config, severity_class_ids
from fire_severity.data.dataset import FireSeverityPatchDataset, collate_batch
from fire_severity.data.encoding import onehot_argmax
from fire_severity.interpretability.analysis import (
    compare_lulc_composition,
    grad_cam_spatial,
    plot_composition_by_severity,
    plot_lulc_proportions_by_severity,
    plot_patch_triplet,
    rank_patches_by_class,
)
from fire_severity.models.unet import SmallUNet
from fire_severity.training.trainer import predict_loader, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trained model.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/fold_0/best_model.pt")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["interpretability"]
    vcfg = cfg.get("validation", {})
    run_name = vcfg.get("run_name", f"fold_{args.fold}")
    out_dir = Path(icfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SmallUNet(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    device = resolve_device(cfg["training"]["device"])
    model.to(device)

    patch_files = sorted(Path(cfg["data"]["patches_root"]).glob("*_patches.npz"))
    ds = FireSeverityPatchDataset(patch_files)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_batch)

    probs, preds, y_true = predict_loader(model, loader, device)
    meta = ds.metadata

    class_ids = lulc_class_ids(cfg)
    comb_ids = combustible_ids(cfg)
    class_names = {int(k): v for k, v in cfg["severity"]["class_names"].items()}
    lulc_names = {int(k): v for k, v in cfg["lulc"]["classes"].items()}
    sev_classes = severity_class_ids(cfg)
    sev_vmax = max(sev_classes)

    # LULC composition by observed dominant severity at patch center
    x_all = np.stack([ds[i]["x"].numpy() for i in range(len(ds))])
    center_labels = meta["center_severity"].values
    comp_df = compare_lulc_composition(x_all, center_labels, class_ids, comb_ids)
    comp_df.to_csv(out_dir / "patch_landscape_metrics.csv", index=False)

    metric_cols = [
        "combustible_fraction",
        "richness",
        "edge_density",
        "combustible_continuity",
        "fuel_nonfuel_contact",
        "fragmentation_index",
        "shannon_diversity",
    ]
    plot_composition_by_severity(comp_df, metric_cols, out_dir / "composition_by_severity.png")
    plot_lulc_proportions_by_severity(comp_df, lulc_names, out_dir / "lulc_proportions_by_severity.png")

    # High-confidence patches per severity class
    loss_masks = np.stack([ds[i]["loss_mask"].numpy() for i in range(len(ds))])
    high_sev = sev_vmax
    for sev in sev_classes:
        ranked = rank_patches_by_class(
            probs, y_true, loss_masks, meta, sev, top_k=icfg["top_k_patches"]
        )
        ranked.to_csv(out_dir / f"top_patches_class_{sev}.csv", index=False)

        # Example visualizations for top 5
        for plot_rank, (_, row) in enumerate(ranked.head(5).iterrows()):
            idx = int(row["dataset_index"])
            lulc = onehot_argmax(x_all[idx], class_ids)
            plot_patch_triplet(
                lulc,
                y_true[idx],
                preds[idx],
                loss_masks[idx],
                class_ids,
                class_names,
                out_dir / f"example_class_{sev}_rank{plot_rank}.png",
                title=f"Clase {class_names[sev]} — conf={row['confidence']:.2f}",
                sev_vmax=sev_vmax,
            )

            if sev == high_sev and plot_rank < 3:
                cam = grad_cam_spatial(model, ds[idx]["x"], target_class=sev, device=device)
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].imshow(lulc, cmap="tab10")
                ax[0].set_title("LULC")
                ax[1].imshow(cam, cmap="hot")
                ax[1].set_title(f"Grad-CAM ({class_names[sev]})")
                for a in ax:
                    a.axis("off")
                fig.savefig(out_dir / f"gradcam_class{sev}_rank{plot_rank}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

    print(f"Interpretability outputs → {out_dir}")


if __name__ == "__main__":
    main()
