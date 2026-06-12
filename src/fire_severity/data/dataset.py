"""PyTorch Dataset for fire severity patches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FireSeverityPatchDataset(Dataset):
    """Loads pre-generated .npz patch bundles."""

    def __init__(self, patch_files: list[Path]):
        self.records: list[dict] = []
        for path in patch_files:
            data = np.load(path, allow_pickle=True)
            fire_id = str(data["fire_id"])
            x = data["x"]
            y = data["y"]
            mask = data["loss_mask"]
            meta = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
            n = x.shape[0]
            for i in range(n):
                self.records.append(
                    {
                        "fire_id": fire_id,
                        "x": x[i],
                        "y": y[i],
                        "loss_mask": mask[i],
                        "center_severity": int(meta["center_severity"][i]),
                        "row": int(meta["row"][i]),
                        "col": int(meta["col"][i]),
                        "source_file": str(path),
                        "patch_index": i,
                    }
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        return {
            "x": torch.from_numpy(rec["x"].astype(np.float32)),
            "y": torch.from_numpy(rec["y"].astype(np.int64)),
            "loss_mask": torch.from_numpy(rec["loss_mask"].astype(bool)),
            "fire_id": rec["fire_id"],
            "center_severity": rec["center_severity"],
        }

    @property
    def metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fire_id": r["fire_id"],
                    "center_severity": r["center_severity"],
                    "row": r["row"],
                    "col": r["col"],
                    "source_file": r["source_file"],
                    "patch_index": r["patch_index"],
                }
                for r in self.records
            ]
        )


def collate_batch(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "loss_mask": torch.stack([b["loss_mask"] for b in batch]),
        "fire_id": [b["fire_id"] for b in batch],
        "center_severity": torch.tensor([b["center_severity"] for b in batch]),
    }
