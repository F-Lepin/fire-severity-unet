"""Evaluation metrics and plots for CNN patch classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    class_names = class_names or ["low", "high"]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_low": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_low": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_low": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_high": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_high": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_high": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc_high"] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        metrics["roc_auc_high"] = float("nan")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                metrics[f"{key}_{sub_key}"] = sub_val
        else:
            metrics[key] = value

    return pd.DataFrame([metrics])


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    class_names: list[str] | None = None,
) -> None:
    class_names = class_names or ["baja (0)", "alta (1)"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
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
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc = roc_auc_score(y_true, y_prob[:, 1])
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
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="train")
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(log_df["epoch"], log_df["val_acc"], label="val acc")
    axes[1].set_title("Accuracy (validación)")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
