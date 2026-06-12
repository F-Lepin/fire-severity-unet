"""Interpretability for CNN LULC patch classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fire_severity.config import combustible_ids, lulc_class_ids
from fire_severity.cnn.dataset import LULCPatchDataset, collate_cnn_batch
from fire_severity.cnn.model import SmallPatchCNN
from fire_severity.cnn.train import predict_loader
from fire_severity.data.encoding import onehot_argmax
from fire_severity.interpretability.analysis import (
    compare_lulc_composition,
    plot_composition_by_severity,
    plot_lulc_proportions_by_severity,
)
from fire_severity.training.trainer import resolve_device


def plot_lulc_patch(
    lulc: np.ndarray,
    out_path: Path,
    title: str = "",
    class_ids: list[int] | None = None,
) -> None:
    class_ids = class_ids or sorted(np.unique(lulc))
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(lulc, cmap="tab10", vmin=min(class_ids), vmax=max(class_ids))
    ax.set_title(title or "LULC previo al incendio")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def rank_patches_by_binary_class(
    probs: np.ndarray,
    metadata: pd.DataFrame,
    target_class: int,
    top_k: int = 50,
) -> pd.DataFrame:
    scores = probs[:, target_class]
    out = metadata.copy()
    out["confidence"] = scores
    out["predicted_class"] = probs.argmax(axis=1)
    out["target_class"] = target_class
    return out.sort_values("confidence", ascending=False).head(top_k)


def export_example_patches(
    x_onehot: np.ndarray,
    ranked: pd.DataFrame,
    class_ids: list[int],
    out_dir: Path,
    prefix: str,
    n_examples: int = 2,
) -> None:
    for rank in range(min(n_examples, len(ranked))):
        idx = int(ranked.iloc[rank]["dataset_index"])
        lulc = onehot_argmax(x_onehot[idx], class_ids)
        conf = float(ranked.iloc[rank]["confidence"])
        plot_lulc_patch(
            lulc,
            out_dir / f"example_{prefix}_rank{rank}.png",
            title=f"{prefix} rank {rank} (conf={conf:.2f})",
            class_ids=class_ids,
        )


def _pca_2d(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-component PCA via SVD (no sklearn)."""
    x = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    variance = singular_values ** 2
    explained = variance / max(variance.sum(), 1e-8)
    return coords, explained


def plot_embeddings_pca(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    if len(embeddings) < 3:
        return
    coords, explained = _pca_2d(embeddings)
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, name, color in [(0, "baja", "#4C72B0"), (1, "alta", "#C44E52")]:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5, label=name, c=color)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title("Embeddings CNN (PCA)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def extract_embeddings(
    model: SmallPatchCNN,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device)
        emb = model.forward_features(x).cpu().numpy()
        chunks.append(emb)
    return np.concatenate(chunks, axis=0)


def run_interpretability(
    model: SmallPatchCNN,
    patch_files: list[Path],
    cfg: dict,
    out_dir: Path,
    device: torch.device | None = None,
) -> None:
    device = device or resolve_device(cfg["training"]["device"])
    model.to(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_ids = lulc_class_ids(cfg)
    comb_ids = combustible_ids(cfg)
    lulc_names = {int(k): v for k, v in cfg["lulc"]["classes"].items()}
    icfg = cfg.get("interpretability", {})
    top_k = int(icfg.get("top_k_patches", 50))

    ds = LULCPatchDataset(patch_files)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_cnn_batch)
    probs, preds, y_true = predict_loader(model, loader, device)
    meta = ds.metadata.copy()
    meta["predicted_class"] = preds
    meta["prob_low"] = probs[:, 0]
    meta["prob_high"] = probs[:, 1]
    meta.to_csv(out_dir / "patch_metadata.csv", index=False)

    x_all = np.stack([ds[i]["x"].numpy() for i in range(len(ds))])
    comp_df = compare_lulc_composition(x_all, meta["binary_label"].values, class_ids, comb_ids)
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
    plot_composition_by_severity(comp_df, metric_cols, out_dir / "landscape_metrics_by_severity.png")
    plot_lulc_proportions_by_severity(comp_df, lulc_names, out_dir / "composition_by_severity.png")

    ranked_low = rank_patches_by_binary_class(probs, meta, target_class=0, top_k=top_k)
    ranked_high = rank_patches_by_binary_class(probs, meta, target_class=1, top_k=top_k)
    ranked_low.to_csv(out_dir / "top_patches_low_severity.csv", index=False)
    ranked_high.to_csv(out_dir / "top_patches_high_severity.csv", index=False)

    export_example_patches(x_all, ranked_low, class_ids, out_dir, "low_severity", n_examples=2)
    export_example_patches(x_all, ranked_high, class_ids, out_dir, "high_severity", n_examples=2)

    if icfg.get("run_embeddings", True):
        embeddings = extract_embeddings(model, loader, device)
        plot_embeddings_pca(embeddings, y_true, out_dir / "embeddings_pca.png")
