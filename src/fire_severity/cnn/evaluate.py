"""Evaluation metrics and plots for CNN patch classifier (numpy-only, no sklearn)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> np.ndarray:
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = int(np.sum((y_true == true_label) & (y_pred == pred_label)))
    return cm


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int) -> tuple[float, float, float]:
    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(y_score))
    n_pos = len(pos)
    n_neg = len(neg)
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def _roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    return fpr, tpr


def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(int((y_true == 1).sum()), 1)
    return recall, precision


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    class_names = class_names or ["low", "high"]
    p0, r0, f0 = _precision_recall_f1(y_true, y_pred, 0)
    p1, r1, f1 = _precision_recall_f1(y_true, y_pred, 1)
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_low": p0,
        "recall_low": r0,
        "f1_low": f0,
        "precision_high": p1,
        "recall_high": r1,
        "f1_high": f1,
        "macro_f1": (f0 + f1) / 2.0,
        "roc_auc_high": _roc_auc(y_true, y_prob[:, 1]),
        "low_precision": p0,
        "low_recall": r0,
        "low_f1-score": f0,
        "high_precision": p1,
        "high_recall": r1,
        "high_f1-score": f1,
        "low_support": int((y_true == 0).sum()),
        "high_support": int((y_true == 1).sum()),
    }
    return pd.DataFrame([metrics])


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    class_names: list[str] | None = None,
) -> None:
    class_names = class_names or ["baja (0)", "alta (1)"]
    cm = _confusion_matrix(y_true, y_pred, [0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Observado")
    ax.set_title("Matriz de confusión (patches)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr = _roc_curve(y_true, y_prob[:, 1])
    auc = _roc_auc(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"Alta severidad (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    recall, precision = _precision_recall_curve(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva precision-recall (clase alta)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_log(log_df: pd.DataFrame, out_path: Path) -> None:
    ncols = 3 if "val_macro_f1" in log_df.columns else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 2:
        axes = list(axes)
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="train")
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(log_df["epoch"], log_df["val_acc"], label="val acc")
    axes[1].set_title("Accuracy (validación)")
    axes[1].legend()
    if ncols == 3:
        axes[2].plot(log_df["epoch"], log_df["val_macro_f1"], label="macro F1")
        axes[2].plot(log_df["epoch"], log_df["val_f1_high"], label="F1 alta")
        axes[2].set_title("F1 validación")
        axes[2].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
