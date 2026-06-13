"""PyTorch dataset for CNN LULC patch classification."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from fire_severity.cnn.augment import augment_lulc_onehot


class LULCPatchDataset(Dataset):
    """Loads one or more CNN patch NPZ bundles (X: one-hot LULC, y: binary label)."""

    def __init__(
        self,
        patch_files: list[Path],
        augment: dict | None = None,
        seed: int | None = None,
    ) -> None:
        self.patch_files = [Path(p) for p in patch_files]
        self.augment = augment or {"enabled": False}
        self._rng = random.Random(seed)
        self._x: list[np.ndarray] = []
        self._y: list[int] = []
        self._meta_rows: list[dict] = []
        self._file_index: list[int] = []

        for file_idx, path in enumerate(self.patch_files):
            data = np.load(path, allow_pickle=True)
            x = data["x"]
            y = data["y"]
            meta_raw = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
            meta_df = pd.DataFrame(meta_raw)
            n = len(y)
            for i in range(n):
                self._x.append(x[i])
                self._y.append(int(y[i]))
                row = meta_df.iloc[i].to_dict()
                row["source_file"] = str(path)
                row["patch_index_in_file"] = i
                row["dataset_index"] = len(self._y) - 1
                self._meta_rows.append(row)
                self._file_index.append(file_idx)

        self.metadata = pd.DataFrame(self._meta_rows)

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int) -> dict:
        x = torch.from_numpy(self._x[idx])
        if self.augment.get("enabled", False):
            x = augment_lulc_onehot(x, self.augment, self._rng)
        return {
            "x": x,
            "y": torch.tensor(self._y[idx], dtype=torch.long),
            "idx": idx,
        }


def collate_cnn_batch(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
    }


def compute_binary_class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss."""
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Per-sample weights for WeightedRandomSampler (inverse class frequency)."""
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weights = 1.0 / counts
    return class_weights[labels.astype(int)]
