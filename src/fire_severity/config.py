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


def severity_class_ids(config: dict[str, Any]) -> list[int]:
    """Model severity class ids (1..N), excluding 0=outside."""
    if "class_names" in config.get("severity", {}):
        return sorted(int(k) for k in config["severity"]["class_names"] if int(k) > 0)
    if "class_ranges" in config.get("severity", {}):
        return sorted(int(k) for k in config["severity"]["class_ranges"])
    return [1, 2, 3]


def severity_class_ranges(config: dict[str, Any]) -> dict[int, tuple[float, float]]:
    raw = config["severity"]["class_ranges"]
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}
