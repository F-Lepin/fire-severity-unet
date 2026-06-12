"""Interpretability: high-confidence patches, Grad-CAM, composition comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fire_severity.data.encoding import onehot_argmax
from fire_severity.interpretability.landscape_metrics import summarize_patch
from fire_severity.models.unet import SmallUNet


def patch_confidence(probs: np.ndarray, target_class: int, loss_mask: np.ndarray) -> float:
    """Mean predicted probability of target_class over valid scar pixels."""
    if probs.ndim == 4:
        probs = probs[0]
    valid = loss_mask
    if valid.sum() == 0:
        return 0.0
    return float(probs[target_class][valid].mean())


def rank_patches_by_class(
    probs: np.ndarray,
    y_true: np.ndarray,
    loss_masks: np.ndarray,
    metadata: pd.DataFrame,
    severity_class: int,
    top_k: int = 50,
) -> pd.DataFrame:
    scores = []
    for i in range(len(probs)):
        score = patch_confidence(probs[i], severity_class, loss_masks[i])
        scores.append(score)
    out = metadata.copy()
    out["dataset_index"] = np.arange(len(probs))
    out["confidence"] = scores
    out["target_class"] = severity_class
    return out.sort_values("confidence", ascending=False).head(top_k)


def grad_cam_spatial(
    model: SmallUNet,
    x: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """
    Simple Grad-CAM on the last encoder feature map (enc3 or enc2).
    Returns H×W importance in [0, 1].
    """
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, out):
        activations["feat"] = out

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["feat"] = grad_out[0]

    hook_layer = model.enc3 if model.enc3 is not None else model.enc2
    h1 = hook_layer.register_forward_hook(fwd_hook)
    h2 = hook_layer.register_full_backward_hook(bwd_hook)

    x = x.to(device)
    x.requires_grad_(True)
    logits = model(x.unsqueeze(0))
    score = logits[0, target_class].mean()
    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    feat = activations["feat"]
    grad = gradients["feat"]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def compare_lulc_composition(
    x_onehot: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
    combustible_ids: set[int],
) -> pd.DataFrame:
    rows = []
    for i in range(len(x_onehot)):
        lulc = onehot_argmax(x_onehot[i], class_ids)
        metrics = summarize_patch(lulc, class_ids, combustible_ids)
        metrics["severity_label"] = int(labels[i])
        rows.append(metrics)
    return pd.DataFrame(rows)


def plot_patch_triplet(
    lulc: np.ndarray,
    severity: np.ndarray,
    pred: np.ndarray,
    loss_mask: np.ndarray,
    class_ids: list[int],
    class_names: dict[int, str],
    out_path: Path,
    title: str = "",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(lulc, cmap="tab10", vmin=min(class_ids), vmax=max(class_ids))
    axes[0].set_title("LULC previo")
    sev_display = np.where(loss_mask, severity, np.nan)
    axes[1].imshow(sev_display, cmap="YlOrRd", vmin=1, vmax=3)
    axes[1].set_title("Severidad observada")
    pred_display = np.where(loss_mask, pred, np.nan)
    axes[2].imshow(pred_display, cmap="YlOrRd", vmin=1, vmax=3)
    axes[2].set_title("Predicción")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_composition_by_severity(
    df: pd.DataFrame,
    metric_cols: list[str],
    out_path: Path,
) -> None:
    grouped = df.groupby("severity_label")[metric_cols].mean()
    grouped.plot(kind="bar", figsize=(10, 4), rot=0)
    plt.ylabel("Valor medio en patch")
    plt.xlabel("Clase de severidad")
    plt.title("Composición LULC por severidad (patches)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_history(history, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.train_loss, label="train")
    ax[0].plot(history.val_loss, label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history.val_acc, label="val acc")
    ax[1].set_title("Accuracy (validación)")
    ax[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close()
