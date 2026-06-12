"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def lulc_class_ids(config: dict[str, Any]) -> list[int]:
    return sorted(int(k) for k in config["lulc"]["classes"])


def num_lulc_channels(config: dict[str, Any]) -> int:
    return len(config["lulc"]["classes"])


def combustible_ids(config: dict[str, Any]) -> set[int]:
    return {int(v) for v in config["lulc"]["combustible_classes"]}
