"""Landscape metrics computed on LULC patches."""

from __future__ import annotations

import numpy as np


def class_proportions(lulc: np.ndarray, class_ids: list[int]) -> dict[int, float]:
    flat = lulc.ravel()
    total = max(len(flat), 1)
    return {cid: float((flat == cid).sum()) / total for cid in class_ids}


def richness(lulc: np.ndarray, class_ids: list[int]) -> int:
    present = set(int(v) for v in np.unique(lulc))
    return len(present & set(class_ids))


def shannon_diversity(lulc: np.ndarray, class_ids: list[int]) -> float:
    props = class_proportions(lulc, class_ids)
    values = [p for p in props.values() if p > 0]
    if not values:
        return 0.0
    return float(-sum(p * np.log(p) for p in values))


def edge_density(lulc: np.ndarray) -> float:
    """Proportion of horizontal/vertical neighbors with different class."""
    if lulc.ndim != 2:
        raise ValueError("Expected 2D LULC array.")
    h, w = lulc.shape
    if h < 2 or w < 2:
        return 0.0
    diff_h = lulc[:, :-1] != lulc[:, 1:]
    diff_v = lulc[:-1, :] != lulc[1:, :]
    edges = diff_h.sum() + diff_v.sum()
    max_edges = (h * (w - 1)) + ((h - 1) * w)
    return float(edges) / max(max_edges, 1)


def combustible_fraction(lulc: np.ndarray, combustible_ids: set[int]) -> float:
    return float(np.isin(lulc, list(combustible_ids)).mean())


def combustible_continuity(lulc: np.ndarray, combustible_ids: set[int]) -> float:
    """
    Mean size of connected combustible components / total combustible pixels.
    Higher values indicate larger contiguous fuel patches.
    """
    mask = np.isin(lulc, list(combustible_ids))
    if not mask.any():
        return 0.0
    visited = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            size = 0
            while stack:
                cr, cc = stack.pop()
                if visited[cr, cc] or not mask[cr, cc]:
                    continue
                visited[cr, cc] = True
                size += 1
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        stack.append((nr, nc))
            sizes.append(size)
    if not sizes:
        return 0.0
    return float(max(sizes)) / float(mask.sum())


def combustible_noncombustible_contact(
    lulc: np.ndarray,
    combustible_ids: set[int],
) -> float:
    """Share of edges that separate combustible from non-combustible cover."""
    comb = np.isin(lulc, list(combustible_ids))
    h, w = comb.shape
    if h < 2 or w < 2:
        return 0.0
    contact_h = comb[:, :-1] != comb[:, 1:]
    contact_v = comb[:-1, :] != comb[1:, :]
    return float(contact_h.sum() + contact_v.sum()) / (
        (h * (w - 1)) + ((h - 1) * w)
    )


def fragmentation_index(lulc: np.ndarray, class_ids: list[int]) -> float:
    """
    Simple fragmentation: number of connected components / richness.
    """
    rich = richness(lulc, class_ids)
    if rich == 0:
        return 0.0
    components = 0
    for cid in class_ids:
        mask = lulc == cid
        if not mask.any():
            continue
        visited = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or visited[r, c]:
                    continue
                components += 1
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if visited[cr, cc] or not mask[cr, cc]:
                        continue
                    visited[cr, cc] = True
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            stack.append((nr, nc))
    return float(components) / rich


def summarize_patch(lulc: np.ndarray, class_ids: list[int], combustible_ids: set[int]) -> dict:
    props = class_proportions(lulc, class_ids)
    return {
        **{f"prop_{cid}": props[cid] for cid in class_ids},
        "richness": richness(lulc, class_ids),
        "shannon_diversity": shannon_diversity(lulc, class_ids),
        "edge_density": edge_density(lulc),
        "combustible_fraction": combustible_fraction(lulc, combustible_ids),
        "combustible_continuity": combustible_continuity(lulc, combustible_ids),
        "fuel_nonfuel_contact": combustible_noncombustible_contact(lulc, combustible_ids),
        "fragmentation_index": fragmentation_index(lulc, class_ids),
    }
