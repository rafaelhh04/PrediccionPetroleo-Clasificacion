"""
Sistema de evaluación común para los 4 modelos del proyecto.

Métricas calculadas (las 5 estándar del proyecto):
    accuracy, precision (macro), recall (macro), f1 (macro), auc_roc

Toda métrica y plot pasa por aquí para garantizar que los 4 modelos
sean directamente comparables (misma definición de macro, mismo umbral
de AUC, etc.).
"""

import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # backend sin display, evita bloquear la pipeline
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Calcula las 5 métricas del proyecto sobre un único conjunto.

    Parameters
    ----------
    y_true : etiquetas reales (0/1)
    y_pred : predicciones binarias (0/1)
    y_proba: probabilidades P(y=1) — necesarias para AUC-ROC

    Returns
    -------
    dict con keys: accuracy, precision, recall, f1, auc_roc
    """
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, y_proba)),
    }


def print_full_metrics(metrics: dict, set_name: str, model_name: str) -> None:
    """
    Vuelca las 5 métricas de un conjunto (train/val/test) con formato legible.
    """
    print(f"[{model_name} — {set_name}]")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}   (macro)")
    print(f"  Recall     : {metrics['recall']:.4f}   (macro)")
    print(f"  F1         : {metrics['f1']:.4f}   (macro)")
    print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: Path,
) -> str:
    """
    Genera y guarda la matriz de confusión como PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Baja (0)", "Sube (1)"])
    ax.set_yticklabels(["Baja (0)", "Sube (1)"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {model_name}")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"confusion_{_safe(model_name)}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_dir: Path,
) -> str:
    """
    Genera y guarda la curva ROC como PNG.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title(f"Curva ROC — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"roc_{_safe(model_name)}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc_comparison(results: list, y_val: np.ndarray, save_dir: Path) -> str:
    """
    Genera una curva ROC con TODOS los modelos superpuestos para comparar visualmente.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_val, r["y_proba_val"])
        auc = r["metrics_val"]["auc_roc"]
        ax.plot(fpr, tpr, linewidth=2, label=f"{r['model_name']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Comparación ROC — todos los modelos (val)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "roc_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def print_summary_table(results: list) -> None:
    """
    Imprime una tabla comparativa de los modelos en consola.
    """
    print("\n" + "=" * 82)
    print(
        f"{'Modelo':<25}"
        f"{'Acc(tr)':>10}{'Acc(val)':>10}{'Prec(val)':>11}"
        f"{'Rec(val)':>10}{'F1(val)':>9}{'AUC(val)':>10}"
    )
    print("-" * 82)
    for r in results:
        m_tr = r["metrics_train"]
        m_va = r["metrics_val"]
        print(
            f"{r['model_name']:<25}"
            f"{m_tr['accuracy']:>10.4f}"
            f"{m_va['accuracy']:>10.4f}"
            f"{m_va['precision']:>11.4f}"
            f"{m_va['recall']:>10.4f}"
            f"{m_va['f1']:>9.4f}"
            f"{m_va['auc_roc']:>10.4f}"
        )
    print("=" * 82 + "\n")


def _safe(name: str) -> str:
    """Convierte 'Logistic Regression' → 'logistic_regression' para nombres de archivo."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")
