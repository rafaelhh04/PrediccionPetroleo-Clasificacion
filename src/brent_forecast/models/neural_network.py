"""
Red neuronal MLP implementada desde cero en NumPy puro.

Arquitectura por defecto: 10 → 64 (ReLU + dropout 0.2) → 32 (ReLU + dropout 0.2) → 1 (sigmoid).
En Fase 5 la arquitectura, lr y dropout son parametrizables vía args de
`train_mlp` y se hace búsqueda manual con TimeSeriesSplit.

Restricciones del proyecto:
- Cero imports de torch, tensorflow, keras o sklearn.neural_network.
- sklearn solo aparece para métricas (roc_auc_score) o vía utils.evaluation.
- random_state=42 propagado por toda operación aleatoria (init, dropout, shuffle).

Notas matemáticas clave (ver CLAUDE_CODE_PLAN_FASE4.md):
- Init He para ReLU: W ~ N(0, sqrt(2/n_in)). Xavier para sigmoid de salida.
- Gradiente combinado sigmoide+BCE: dZ3 = (A3 - y) / m (numéricamente estable).
- Inverted dropout: scale por 1/keep en train; eval es identidad.
- BCE con clipping a [1e-15, 1-1e-15] para evitar log(0).
"""

import os
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from brent_forecast.evaluation.metrics import (
    compute_metrics,
    print_full_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from brent_forecast.config import ModelSettings
from brent_forecast.models.tuning import print_grid_results


MODEL_NAME = "MLP NumPy"

# Hyperparámetros aceptados por `train_mlp` (los valores vienen de la config).
TRAIN_KWARGS = frozenset({
    "hidden_1", "hidden_2", "dropout_p", "learning_rate",
    "batch_size", "max_epochs", "patience", "random_state",
})


# ──────────────────────────────────────────────
# 1. Funciones de activación y derivadas
# ──────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -500.0, 500.0)
    pos = 1.0 / (1.0 + np.exp(-z_clip))
    neg = np.exp(z_clip) / (1.0 + np.exp(z_clip))
    return np.where(z >= 0, pos, neg)


# ──────────────────────────────────────────────
# 2. Inicialización (He / Xavier)
# ──────────────────────────────────────────────

def initialize_parameters(
    n_in: int,
    n_h1: int,
    n_h2: int,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    return {
        "W1": rng.randn(n_in, n_h1) * np.sqrt(2.0 / n_in),
        "b1": np.zeros(n_h1),
        "W2": rng.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1),
        "b2": np.zeros(n_h2),
        "W3": rng.randn(n_h2, 1)    * np.sqrt(1.0 / n_h2),
        "b3": np.zeros(1),
    }


# ──────────────────────────────────────────────
# 3. Forward pass (con inverted dropout en train)
# ──────────────────────────────────────────────

def forward(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    training: bool,
    rng: np.random.RandomState,
    dropout_p: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Si training=True aplica inverted dropout con prob `dropout_p`.
    Si training=False dropout es identidad — red determinista.
    """
    keep = 1.0 - dropout_p

    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    if training:
        D1 = (rng.rand(*A1.shape) < keep).astype(A1.dtype)
        A1 = A1 * D1 / keep
    else:
        D1 = None

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    if training:
        D2 = (rng.rand(*A2.shape) < keep).astype(A2.dtype)
        A2 = A2 * D2 / keep
    else:
        D2 = None

    Z3 = A2 @ params["W3"] + params["b3"]
    A3 = sigmoid(Z3)

    cache = {
        "X": X, "Z1": Z1, "A1": A1, "D1": D1,
        "Z2": Z2, "A2": A2, "D2": D2,
        "Z3": Z3, "A3": A3,
    }
    return A3, cache


# ──────────────────────────────────────────────
# 4. Loss: binary cross-entropy
# ──────────────────────────────────────────────

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    y_pred = np.clip(y_pred.ravel(), eps, 1.0 - eps)
    y_true = y_true.ravel().astype(np.float64)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


# ──────────────────────────────────────────────
# 5. Backward pass
# ──────────────────────────────────────────────

def backward(
    y_true: np.ndarray,
    cache: Dict[str, np.ndarray],
    params: Dict[str, np.ndarray],
    dropout_p: float,
) -> Dict[str, np.ndarray]:
    keep = 1.0 - dropout_p
    m = y_true.shape[0]
    y = y_true.reshape(-1, 1).astype(np.float64)

    dZ3 = (cache["A3"] - y) / m
    dW3 = cache["A2"].T @ dZ3
    db3 = dZ3.sum(axis=0)

    dA2 = dZ3 @ params["W3"].T
    if cache["D2"] is not None:
        dA2 = dA2 * cache["D2"] / keep
    dZ2 = dA2 * relu_deriv(cache["Z2"])
    dW2 = cache["A1"].T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ params["W2"].T
    if cache["D1"] is not None:
        dA1 = dA1 * cache["D1"] / keep
    dZ1 = dA1 * relu_deriv(cache["Z1"])
    dW1 = cache["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}


# ──────────────────────────────────────────────
# 6. SGD step
# ──────────────────────────────────────────────

def sgd_step(
    params: Dict[str, np.ndarray],
    grads: Dict[str, np.ndarray],
    lr: float,
) -> Dict[str, np.ndarray]:
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        params[key] = params[key] - lr * grads[f"d{key}"]
    return params


# ──────────────────────────────────────────────
# 7. Training loop parametrizado (Fase 5)
# ──────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    *,
    hidden_1:      int,
    hidden_2:      int,
    dropout_p:     float,
    learning_rate: float,
    batch_size:    int,
    max_epochs:    int,
    patience:      int,
    random_state:  int,
    verbose:       bool  = True,
) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
    """
    Entrena con SGD mini-batch + shuffle por epoch + early stopping sobre val AUC.
    Todos los hyperparámetros (incluida la semilla) llegan como argumentos.
    `verbose=False` silencia los logs por epoch — útil durante el tuning.
    """
    from sklearn.metrics import roc_auc_score  # uso permitido: solo monitorización

    rng   = np.random.RandomState(random_state)
    n_in  = X_train.shape[1]
    n     = X_train.shape[0]

    params = initialize_parameters(n_in, hidden_1, hidden_2, rng)

    best_auc          = -np.inf
    best_params       = {k: v.copy() for k, v in params.items()}
    best_epoch        = 0
    epochs_no_improve = 0
    history: List[Dict] = []

    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_losses = []
        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]

            A3, cache = forward(Xb, params, training=True, rng=rng, dropout_p=dropout_p)
            loss      = binary_cross_entropy(yb, A3)
            grads     = backward(yb, cache, params, dropout_p=dropout_p)
            params    = sgd_step(params, grads, learning_rate)
            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses))
        val_proba  = predict_proba(X_val, params)
        val_loss   = binary_cross_entropy(y_val, val_proba)
        val_auc    = float(roc_auc_score(y_val, val_proba))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_auc":    val_auc,
        })

        improved = val_auc > best_auc
        if improved:
            best_auc          = val_auc
            best_params       = {k: v.copy() for k, v in params.items()}
            best_epoch        = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            flag = " ★" if improved else ""
            if epoch == 1 or epoch % 5 == 0 or improved:
                print(f"[MLP] epoch {epoch:>3} | train_loss {train_loss:.4f} | "
                      f"val_loss {val_loss:.4f} | val_auc {val_auc:.4f}{flag}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"[MLP] Early stopping en epoch {epoch} (sin mejora durante {patience} epochs).")
            break

    if verbose:
        print(f"[MLP] Mejor AUC val: {best_auc:.4f} (epoch {best_epoch})")
    return best_params, history


# ──────────────────────────────────────────────
# 8. Predicción
# ──────────────────────────────────────────────

def predict_proba(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Forward con training=False; devuelve P(y=1) como vector 1D.
    En inferencia el dropout es la identidad, así que no depende de dropout_p.
    """
    dummy_rng = np.random.RandomState(0)
    A3, _ = forward(X, params, training=False, rng=dummy_rng, dropout_p=0.0)
    return A3.ravel()


def predict(
    X: np.ndarray,
    params: Dict[str, np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray:
    return (predict_proba(X, params) >= threshold).astype(int)


# ──────────────────────────────────────────────
# 9. Hyperparameter tuning (búsqueda manual sobre TimeSeriesSplit)
# ──────────────────────────────────────────────

def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv,
    config: ModelSettings,
    scoring: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Búsqueda manual sobre `config.grid` (hidden_pair × learning_rate) usando TimeSeriesSplit.
    Para cada combinación, entrena en cada fold y computa AUC val medio.

    No usa GridSearchCV porque la MLP no es sklearn-compatible; hacer un
    wrapper BaseEstimator añadiría complejidad innecesaria.
    """
    from sklearn.metrics import roc_auc_score

    if scoring != "roc_auc":
        raise ValueError(f"{MODEL_NAME} tuning only supports scoring='roc_auc', got {scoring!r}")

    default_params = {**config.params, "random_state": seed}
    base_kwargs = {k: v for k, v in default_params.items() if k in TRAIN_KWARGS}
    hidden_pairs = config.grid["hidden_pair"]
    learning_rates = config.grid["learning_rate"]
    results = []

    print(f"\n[{MODEL_NAME}] Tuning manual: {len(hidden_pairs) * len(learning_rates)} "
          f"combos × {cv.get_n_splits()} folds...")

    for (h1, h2), lr in product(hidden_pairs, learning_rates):
        fold_aucs = []
        for tr_idx, va_idx in cv.split(X_train):
            X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
            X_va_f, y_va_f = X_train[va_idx], y_train[va_idx]

            best_params, _ = train_mlp(
                X_tr_f, y_tr_f, X_va_f, y_va_f,
                **{**base_kwargs, "hidden_1": h1, "hidden_2": h2, "learning_rate": lr},
                verbose=False,
            )
            y_proba = predict_proba(X_va_f, best_params)
            fold_aucs.append(float(roc_auc_score(y_va_f, y_proba)))

        params_dict = {"hidden_1": h1, "hidden_2": h2, "learning_rate": lr}
        mean_auc = float(np.mean(fold_aucs))
        std_auc  = float(np.std(fold_aucs))
        results.append({
            "params":      params_dict,
            "mean_cv_auc": mean_auc,
            "std_cv_auc":  std_auc,
        })
        print(f"[{MODEL_NAME}]   (H1={h1}, H2={h2}, lr={lr}) → "
              f"AUC CV = {mean_auc:.4f} ± {std_auc:.4f}")

    print_grid_results(MODEL_NAME, results, top_k=5)
    best = max(results, key=lambda r: r["mean_cv_auc"])
    return {**default_params, **best["params"]}


# ──────────────────────────────────────────────
# 10. Wrapper público (contrato Fase 3)
# ──────────────────────────────────────────────

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    feature_cols: List[str],
    params: Dict[str, Any],
    plots_dir: Path,
) -> Dict:
    """
    Cumple el contrato común de Fase 3.
    """
    print(f"\n[{MODEL_NAME}] Entrenando con params: {params}...")

    # Filtrar solo los kwargs aceptados por train_mlp
    train_kwargs = {k: v for k, v in params.items() if k in TRAIN_KWARGS}
    best_params, history = train_mlp(X_train, y_train, X_val, y_val, **train_kwargs)

    y_proba_train = predict_proba(X_train, best_params)
    y_proba_val   = predict_proba(X_val,   best_params)
    y_pred_train  = (y_proba_train >= 0.5).astype(int)
    y_pred_val    = (y_proba_val   >= 0.5).astype(int)

    metrics_train = compute_metrics(y_train, y_pred_train, y_proba_train)
    metrics_val   = compute_metrics(y_val,   y_pred_val,   y_proba_val)

    print_full_metrics(metrics_train, "Train", MODEL_NAME)
    print_full_metrics(metrics_val,   "Val",   MODEL_NAME)
    print(f"[{MODEL_NAME}] Epochs entrenados: {len(history)} | "
          f"Mejor AUC val: {max(h['val_auc'] for h in history):.4f}")

    plot_confusion_matrix(y_val, y_pred_val, MODEL_NAME, plots_dir)
    plot_roc_curve(y_val, y_proba_val, MODEL_NAME, plots_dir)
    _plot_training_curves(history, plots_dir)

    return {
        "model_name":    MODEL_NAME,
        "model":         {"params": best_params, "history": history, "config": params},
        "metrics_train": metrics_train,
        "metrics_val":   metrics_val,
        "y_pred_val":    y_pred_val,
        "y_proba_val":   y_proba_val,
    }


def _plot_training_curves(history: List[Dict], save_dir: Path) -> str:
    """Guarda la evolución de train_loss / val_loss / val_auc por epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs   = [h["epoch"]      for h in history]
    train_l  = [h["train_loss"] for h in history]
    val_l    = [h["val_loss"]   for h in history]
    val_aucs = [h["val_auc"]    for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_l, label="train_loss")
    axes[0].plot(epochs, val_l,   label="val_loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Curva de pérdida — MLP NumPy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_aucs, color="C2", label="val_auc")
    axes[1].axhline(0.5, linestyle="--", color="gray", label="Aleatorio")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC val")
    axes[1].set_title("AUC val por epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "training_curves_mlp_numpy.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
