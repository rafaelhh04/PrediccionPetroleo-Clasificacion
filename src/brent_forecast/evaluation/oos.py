"""Out-of-sample (walk-forward) metrics, tables and figures."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from brent_forecast._types import Metrics
from brent_forecast.evaluation.metrics import compute_metrics, save_figure

logger = logging.getLogger(__name__)


class WindowMetrics(TypedDict):
    """Metrics of one walk-forward window."""

    window: int
    start: str
    end: str
    n: int
    auc_roc: float | None
    accuracy: float


def oos_metrics(y: pd.Series, proba: pd.Series, threshold: float = 0.5) -> Metrics:
    """Compute the five project metrics over all out-of-sample predictions."""
    return compute_metrics(y, (proba >= threshold).astype(int), proba)


def window_metrics(
    dates: pd.Series, y: pd.Series, proba: pd.Series, window: pd.Series
) -> list[WindowMetrics]:
    """Metrics of every walk-forward window (AUC is ``None`` if a window has one class)."""
    rows: list[WindowMetrics] = []
    for k in sorted(window.unique()):
        idx = window.index[window == k]
        y_k, p_k = y.loc[idx], proba.loc[idx]
        auc = float(roc_auc_score(y_k, p_k)) if y_k.nunique() == 2 else None
        rows.append(
            {
                "window": int(k),
                "start": str(dates.loc[idx].min().date()),
                "end": str(dates.loc[idx].max().date()),
                "n": len(idx),
                "auc_roc": auc,
                "accuracy": float(accuracy_score(y_k, (p_k >= 0.5).astype(int))),
            }
        )
    return rows


def log_oos_summary(summary: Mapping[str, Mapping[str, Any]]) -> None:
    """Log the development-CV vs out-of-sample comparison table."""
    width = 25 + 17 + 10 + 10 + 14 + 16
    lines = [
        "=" * width,
        f"{'Model':<25}{'CV AUC (dev)':>17}{'OOS AUC':>10}{'OOS acc':>10}"
        f"{'win AUC mean':>14}{'win AUC range':>16}",
        "-" * width,
    ]
    for name, m in summary.items():
        aucs = [w["auc_roc"] for w in m["windows"] if w["auc_roc"] is not None]
        cv = f"{m['cv']['auc_mean']:.4f} ± {m['cv']['auc_std']:.3f}"
        label = f"{name} *" if m.get("kind") == "baseline" else name
        lines.append(
            f"{label:<25}{cv:>17}{m['oos']['auc_roc']:>10.4f}{m['oos']['accuracy']:>10.4f}"
            f"{np.mean(aucs):>14.4f}{f'{min(aucs):.3f}-{max(aucs):.3f}':>16}"
        )
    lines.append("=" * width)
    lines.append("* naive baseline")
    logger.info("Walk-forward summary:\n%s", "\n".join(lines))


def plot_roc_oos(y: pd.Series, probas: Mapping[str, pd.Series], plots_dir: Path) -> Path:
    """ROC curves of every model on the pooled out-of-sample predictions."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("True positive rate (TPR)")
    ax.set_title("Out-of-sample ROC (walk-forward)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, plots_dir, "roc_oos_comparison.png")


def plot_window_auc(windows: Mapping[str, list[WindowMetrics]], plots_dir: Path) -> Path:
    """AUC of every model per walk-forward window (stability over time)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, rows in windows.items():
        starts = pd.to_datetime([w["start"] for w in rows]).to_numpy()
        aucs = [np.nan if w["auc_roc"] is None else w["auc_roc"] for w in rows]
        ax.plot(starts, aucs, "o-", label=name)
    ax.axhline(0.5, linestyle="--", color="gray", label="Random")
    ax.set_xlabel("Window start")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Out-of-sample AUC per walk-forward window")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_figure(fig, plots_dir, "oos_auc_per_window.png")
