"""Training loop for CNN LULC patch classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from fire_severity.cnn.dataset import (
    LULCPatchDataset,
    collate_cnn_batch,
    compute_binary_class_weights,
    compute_sample_weights,
)
from fire_severity.cnn.model import SmallPatchCNN
from fire_severity.training.trainer import resolve_device


@dataclass
class CNNTrainConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    checkpoint_dir: Path = Path("checkpoints_cnn_lulc_binary")
    use_class_weights: bool = True
    use_weighted_sampler: bool = True
    checkpoint_metric: str = "macro_f1"
    early_stopping_patience: int = 10
    min_epochs: int = 5
    augment: dict = field(default_factory=lambda: {"enabled": False})
    seed: int = 42


@dataclass
class CNNTrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    val_f1_low: list[float] = field(default_factory=list)
    val_f1_high: list[float] = field(default_factory=list)


def _f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> list[float]:
    scores: list[float] = []
    for cls in range(num_classes):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return scores


def run_epoch(
    model: SmallPatchCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float, float, float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0
    all_preds: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_preds)
    f1_scores = _f1_per_class(y_true, y_pred)
    macro_f1 = float(np.mean(f1_scores))
    acc = float(np.mean(y_true == y_pred))
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, acc, macro_f1, f1_scores[0], f1_scores[1]


def build_train_loader(ds: LULCPatchDataset, cfg: CNNTrainConfig) -> DataLoader:
    if cfg.use_weighted_sampler:
        labels = ds.metadata["binary_label"].values.astype(int)
        sample_weights = compute_sample_weights(labels)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(labels),
            replacement=True,
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate_cnn_batch,
        )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_cnn_batch,
    )


def _metric_improved(metric_name: str, current: float, best: float) -> bool:
    if metric_name == "val_loss":
        return current < best
    return current > best


def train_cnn_model(
    train_files: list[Path],
    val_files: list[Path],
    model_cfg: dict,
    train_cfg: CNNTrainConfig,
) -> tuple[SmallPatchCNN, CNNTrainHistory, pd.DataFrame, dict]:
    device = resolve_device(train_cfg.device)
    train_ds = LULCPatchDataset(train_files, augment=train_cfg.augment, seed=train_cfg.seed)
    val_ds = LULCPatchDataset(val_files)

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty.")

    class_weights = compute_binary_class_weights(train_ds.metadata["binary_label"].values)
    if train_cfg.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    train_loader = build_train_loader(train_ds, train_cfg)
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_cnn_batch,
    )

    model = SmallPatchCNN(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg.get("num_classes", 2),
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    metric_name = train_cfg.checkpoint_metric
    history = CNNTrainHistory()
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = train_cfg.checkpoint_dir / "model_best.pt"

    if metric_name == "val_loss":
        best_score = float("inf")
    else:
        best_score = float("-inf")

    best_meta: dict = {}
    epochs_without_improvement = 0

    for epoch in range(train_cfg.epochs):
        tr_loss, tr_acc, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_macro_f1, va_f1_low, va_f1_high = run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)
        history.val_macro_f1.append(va_macro_f1)
        history.val_f1_low.append(va_f1_low)
        history.val_f1_high.append(va_f1_high)

        if metric_name == "val_loss":
            score = va_loss
        elif metric_name == "val_acc":
            score = va_acc
        else:
            score = va_macro_f1

        if _metric_improved(metric_name, score, best_score):
            best_score = score
            epochs_without_improvement = 0
            best_meta = {
                "model_state": model.state_dict(),
                "model_cfg": model_cfg,
                "epoch": epoch,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "val_macro_f1": va_macro_f1,
                "val_f1_low": va_f1_low,
                "val_f1_high": va_f1_high,
                "checkpoint_metric": metric_name,
                "checkpoint_score": score,
            }
            torch.save(best_meta, best_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{train_cfg.epochs} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
            f"macro_f1={va_macro_f1:.4f} f1_high={va_f1_high:.4f}"
        )

        if epoch + 1 >= train_cfg.min_epochs and epochs_without_improvement >= train_cfg.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no improvement in {metric_name} for {train_cfg.early_stopping_patience} epochs)."
            )
            break

    if not best_path.exists():
        raise RuntimeError("No checkpoint was saved during training.")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(
        f"Loaded best checkpoint from epoch {ckpt['epoch'] + 1} "
        f"({metric_name}={ckpt.get('checkpoint_score', best_score):.4f})"
    )

    log_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.train_loss) + 1),
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_acc": history.val_acc,
            "val_macro_f1": history.val_macro_f1,
            "val_f1_low": history.val_f1_low,
            "val_f1_high": history.val_f1_high,
        }
    )
    return model, history, log_df, ckpt


@torch.no_grad()
def predict_loader(
    model: SmallPatchCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_y.append(y)

    return np.concatenate(all_probs), np.concatenate(all_preds), np.concatenate(all_y)
