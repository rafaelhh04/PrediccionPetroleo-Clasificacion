"""
Curvas de aprendizaje con TimeSeriesSplit.

Para sklearn (LogReg/SVM/RF) usa learning_curve nativo con cv=TimeSeriesSplit.
Para MLP NumPy es manual: re-entrena train_mlp sobre subsets crecientes en
cada fold (no es sklearn-compatible).

Todas las curvas se guardan en results/plots/learning_curve_<modelo>.png.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from utils.tuning import make_time_series_cv, SCORING


PLOTS_DIR = "results/plots"
TRAIN_SIZES = np.array([0.1, 0.25, 0.5, 0.75, 1.0])


def plot_learning_curve_sklearn(estimator, X_train, y_train, model_name: str) -> str:
    """
    Curva de aprendizaje para estimator sklearn con TimeSeriesSplit.

    Devuelve la ruta del PNG generado.
    """
    print(f"[learning_curve] Calculando curva para {model_name}...")
    cv = make_time_series_cv()
    sizes, train_scores, val_scores = learning_curve(
        estimator, X_train, y_train,
        train_sizes=TRAIN_SIZES,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
    )
    return _save_learning_curve_plot(sizes, train_scores, val_scores, model_name)


def plot_learning_curve_mlp(X_train, y_train, mlp_params: dict, model_name: str = "MLP NumPy") -> str:
    """
    Curva de aprendizaje manual para MLP NumPy.

    Para cada tamaño relativo, entrena el MLP con esos primeros N samples
    (respetando orden temporal) sobre cada fold de TimeSeriesSplit.
    Reporta media y std de AUC train vs AUC val.
    """
    from sklearn.metrics import roc_auc_score
    from models.neural_network import train_mlp, predict_proba

    print(f"[learning_curve] Calculando curva manual para {model_name}...")
    cv = make_time_series_cv()
    train_scores = []
    val_scores   = []

    train_kwargs = {
        k: v for k, v in mlp_params.items()
        if k in {"hidden_1", "hidden_2", "dropout_p", "learning_rate",
                 "batch_size", "max_epochs", "patience"}
    }

    for frac in TRAIN_SIZES:
        size_train = []
        size_val   = []
        for tr_idx, va_idx in cv.split(X_train):
            n_use = max(50, int(len(tr_idx) * frac))
            tr_sub = tr_idx[:n_use]

            X_tr, y_tr = X_train[tr_sub], y_train[tr_sub]
            X_va, y_va = X_train[va_idx], y_train[va_idx]
            best_params, _ = train_mlp(X_tr, y_tr, X_va, y_va, verbose=False, **train_kwargs)

            size_train.append(roc_auc_score(y_tr, predict_proba(X_tr, best_params)))
            size_val.append(roc_auc_score(y_va, predict_proba(X_va, best_params)))

        train_scores.append(size_train)
        val_scores.append(size_val)
        print(f"[learning_curve]   frac={frac:.2f} → AUC train {np.mean(size_train):.4f} | "
              f"AUC CV {np.mean(size_val):.4f}")

    sizes_abs = (TRAIN_SIZES * len(X_train)).astype(int)
    return _save_learning_curve_plot(
        sizes_abs, np.array(train_scores), np.array(val_scores), model_name
    )


def _save_learning_curve_plot(sizes, train_scores, val_scores, model_name: str) -> str:
    """Plot común: media ± std de AUC train y AUC CV por tamaño de muestra."""
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, train_mean, "o-", color="C0", label="AUC train")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="C0")
    ax.plot(sizes, val_mean, "s-", color="C1", label="AUC CV (val fold)")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="C1")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Aleatorio")
    ax.set_xlabel("Tamaño del subconjunto de train")
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Curva de aprendizaje — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(PLOTS_DIR, f"learning_curve_{safe}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
