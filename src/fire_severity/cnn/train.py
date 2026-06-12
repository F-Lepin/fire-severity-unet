"""Training loop for CNN LULC patch classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from fire_severity.cnn.dataset import LULCPatchDataset, collate_cnn_batch, compute_binary_class_weights
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
    use_weighted_sampler: bool = False


@dataclass
class CNNTrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    if len(targets) == 0:
        return 0.0
    return float((preds == targets).float().mean().item())


def run_epoch(
    model: SmallPatchCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

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
        total_acc += binary_accuracy(logits.detach(), y)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def build_train_loader(
    ds: LULCPatchDataset,
    cfg: CNNTrainConfig,
    class_weights: torch.Tensor | None,
) -> DataLoader:
    if cfg.use_weighted_sampler and not cfg.use_class_weights:
        labels = ds.metadata["binary_label"].values.astype(int)
        sample_weights = class_weights[labels] if class_weights is not None else np.ones(len(labels))
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


def train_cnn_model(
    train_files: list[Path],
    val_files: list[Path],
    model_cfg: dict,
    train_cfg: CNNTrainConfig,
) -> tuple[SmallPatchCNN, CNNTrainHistory, pd.DataFrame]:
    device = resolve_device(train_cfg.device)
    train_ds = LULCPatchDataset(train_files)
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

    cw_for_sampler = class_weights if train_cfg.use_weighted_sampler else None
    train_loader = build_train_loader(train_ds, train_cfg, cw_for_sampler)
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

    history = CNNTrainHistory()
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_path = train_cfg.checkpoint_dir / "model_best.pt"

    for epoch in range(train_cfg.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg,
                    "epoch": epoch,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                },
                best_path,
            )

        print(
            f"Epoch {epoch + 1}/{train_cfg.epochs} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    log_df = pd.DataFrame(
        {
            "epoch": np.arange(1, train_cfg.epochs + 1),
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "val_acc": history.val_acc,
        }
    )
    return model, history, log_df


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
