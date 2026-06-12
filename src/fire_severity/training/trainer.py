"""Training loop for exploratory severity U-Net."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fire_severity.data.dataset import FireSeverityPatchDataset, collate_batch
from fire_severity.models.unet import SmallUNet
from fire_severity.training.loss import MaskedCrossEntropyLoss, compute_class_weights


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    checkpoint_dir: Path = Path("checkpoints")


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    valid = loss_mask & (targets > 0)
    if valid.sum() == 0:
        return 0.0
    correct = (preds[valid] == targets[valid]).float().mean().item()
    return correct


def run_epoch(
    model: SmallUNet,
    loader: DataLoader,
    criterion: MaskedCrossEntropyLoss,
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
        mask = batch["loss_mask"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y, mask)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_acc += masked_accuracy(logits.detach(), y, mask)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train_model(
    train_files: list[Path],
    val_files: list[Path],
    model_cfg: dict,
    train_cfg: TrainConfig,
) -> tuple[SmallUNet, TrainHistory]:
    device = resolve_device(train_cfg.device)
    train_ds = FireSeverityPatchDataset(train_files)
    val_ds = FireSeverityPatchDataset(val_files)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_batch,
    )

    model = SmallUNet(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg.get("base_channels", 32),
        encoder_depth=model_cfg.get("encoder_depth", 2),
    ).to(device)

    sample_labels = torch.stack([train_ds[i]["y"] for i in range(min(64, len(train_ds)))])
    weights = compute_class_weights(sample_labels, model_cfg["num_classes"]).to(device)
    criterion = MaskedCrossEntropyLoss(class_weights=weights).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    history = TrainHistory()
    train_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(train_cfg.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)

        print(
            f"Epoch {epoch + 1}/{train_cfg.epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": model_cfg,
                    "epoch": epoch,
                    "val_loss": va_loss,
                },
                train_cfg.checkpoint_dir / "best_model.pt",
            )

    return model, history


def predict_loader(
    model: SmallUNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].numpy()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_true.append(y)

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_true, axis=0),
    )
