"""
Evaluación final sobre X_test.

REGLA CRÍTICA DEL PROYECTO: este módulo se invoca **una sola vez** desde
main.py. Cualquier iteración del modelo basada en métricas de test es
overfitting al test set y corrompe la honestidad metodológica.

Si las métricas de test salen malas, se reportan tal cual. No se vuelve
atrás a re-tunear.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from brent_forecast.evaluation.metrics import compute_metrics, print_full_metrics, plot_confusion_matrix


PLOTS_DIR = "results/plots"


def evaluate_on_test(fitted_results: list, X_test: np.ndarray, y_test: np.ndarray) -> list:
    """
    Para cada modelo en `fitted_results`, evalúa sobre (X_test, y_test).

    Distingue sklearn vs MLP por la estructura del campo 'model':
    - sklearn → estimator con predict_proba.
    - MLP     → dict {params, history, config}.

    Devuelve una nueva lista con las claves originales más:
    - 'metrics_test': dict con las 5 métricas estándar.
    - 'y_pred_test', 'y_proba_test'.
    """
    from brent_forecast.models.neural_network import predict_proba as mlp_predict_proba

    print("\n" + "=" * 60)
    print("EVALUACIÓN FINAL SOBRE X_TEST")
    print("Una sola pasada — el test set NO se usa para nada más")
    print("=" * 60)

    test_results = []
    for r in fitted_results:
        name  = r["model_name"]
        model = r["model"]

        if isinstance(model, dict) and "params" in model:
            dropout_p = model.get("config", {}).get("dropout_p", 0.2)
            y_proba = mlp_predict_proba(X_test, model["params"], dropout_p=dropout_p)
        else:
            y_proba = model.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= 0.5).astype(int)
        metrics_test = compute_metrics(y_test, y_pred, y_proba)

        print_full_metrics(metrics_test, "Test", name)
        plot_confusion_matrix(y_test, y_pred, f"{name} (test)")

        test_results.append({
            **r,
            "metrics_test": metrics_test,
            "y_pred_test":  y_pred,
            "y_proba_test": y_proba,
        })

    return test_results


def plot_roc_test_comparison(test_results: list, y_test: np.ndarray) -> str:
    """ROC de los 4 modelos superpuestos sobre test."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in test_results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba_test"])
        auc = r["metrics_test"]["auc_roc"]
        ax.plot(fpr, tpr, linewidth=2, label=f"{r['model_name']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatorio")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC final sobre TEST — 4 modelos")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "roc_test_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_train_val_test_summary(test_results: list) -> str:
    """Gráfico de barras: AUC en train/val/test por modelo. Visualiza el shift."""
    names  = [r["model_name"]              for r in test_results]
    auc_tr = [r["metrics_train"]["auc_roc"] for r in test_results]
    auc_va = [r["metrics_val"]["auc_roc"]   for r in test_results]
    auc_te = [r["metrics_test"]["auc_roc"]  for r in test_results]

    x = np.arange(len(names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, auc_tr, width, label="AUC train", color="C0")
    ax.bar(x,         auc_va, width, label="AUC val",   color="C1")
    ax.bar(x + width, auc_te, width, label="AUC test",  color="C2")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Aleatorio")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("AUC en train / val / test — los 4 modelos")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "train_val_test_summary.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def print_final_summary_table(test_results: list) -> None:
    """Tabla comparativa final: AUC train / val / test por modelo."""
    print("\n" + "=" * 78)
    print(f"{'Modelo':<25}{'AUC(tr)':>12}{'AUC(val)':>12}{'AUC(test)':>13}{'Gap te-val':>14}")
    print("-" * 78)
    for r in test_results:
        auc_tr = r["metrics_train"]["auc_roc"]
        auc_va = r["metrics_val"]["auc_roc"]
        auc_te = r["metrics_test"]["auc_roc"]
        gap    = auc_te - auc_va
        print(f"{r['model_name']:<25}{auc_tr:>12.4f}{auc_va:>12.4f}{auc_te:>13.4f}{gap:>+14.4f}")
    print("=" * 78 + "\n")
