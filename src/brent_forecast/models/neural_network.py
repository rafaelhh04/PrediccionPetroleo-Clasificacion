"""Multilayer perceptron implemented from scratch in pure NumPy.

Default architecture: n_in -> 64 (ReLU + dropout 0.2) -> 32 (ReLU + dropout 0.2)
-> 1 (sigmoid). Layer sizes, learning rate and dropout come from the
configuration, and the grid search is done manually with TimeSeriesSplit.

Project constraints:

- No torch, tensorflow, keras or sklearn.neural_network imports.
- sklearn is only used for metrics (``roc_auc_score``).
- The seed (``random_state``) drives every random operation: initialisation,
  dropout masks and shuffling.

Key mathematical notes:

- He initialisation for ReLU layers: ``W ~ N(0, sqrt(2 / n_in))``; Xavier for
  the sigmoid output layer.
- Combined sigmoid + BCE gradient: ``dZ3 = (A3 - y) / m`` (numerically stable).
- Inverted dropout: scale by ``1 / keep`` while training; identity at inference.
- BCE clips predictions to ``[1e-15, 1 - 1e-15]`` to avoid ``log(0)``.
"""

import logging
from itertools import product
from pathlib import Path
from typing import Any, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from brent_forecast._types import FloatArray, IntArray, ModelResult
from brent_forecast.config import ModelSettings
from brent_forecast.evaluation.metrics import (
    compute_metrics,
    log_full_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_figure,
)
from brent_forecast.models.tuning import GridResult, log_grid_results

logger = logging.getLogger(__name__)

MODEL_NAME = "MLP NumPy"

# Hyperparameters accepted by ``train_mlp`` (values come from the configuration).
TRAIN_KWARGS = frozenset(
    {
        "hidden_1",
        "hidden_2",
        "dropout_p",
        "learning_rate",
        "batch_size",
        "max_epochs",
        "patience",
        "random_state",
    }
)

Params = dict[str, FloatArray]
"""Network weights: ``W1, b1, W2, b2, W3, b3``."""


class ForwardCache(TypedDict):
    """Intermediate values of the forward pass needed by backpropagation."""

    X: FloatArray
    Z1: FloatArray
    A1: FloatArray
    D1: FloatArray | None
    Z2: FloatArray
    A2: FloatArray
    D2: FloatArray | None
    Z3: FloatArray
    A3: FloatArray


class EpochRecord(TypedDict):
    """Training history of one epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float


# ──────────────────────────────────────────────
# 1. Activations and derivatives
# ──────────────────────────────────────────────


def relu(z: FloatArray) -> FloatArray:
    """Apply the rectified linear unit."""
    return np.maximum(0.0, z)


def relu_deriv(z: FloatArray) -> FloatArray:
    """Compute the derivative of ReLU (0 at z = 0)."""
    return (z > 0).astype(z.dtype)


def sigmoid(z: FloatArray) -> FloatArray:
    """Compute the logistic sigmoid in a numerically stable way."""
    z_clip = np.clip(z, -500.0, 500.0)
    pos = 1.0 / (1.0 + np.exp(-z_clip))
    neg = np.exp(z_clip) / (1.0 + np.exp(z_clip))
    return np.where(z >= 0, pos, neg)


# ──────────────────────────────────────────────
# 2. Initialisation (He / Xavier)
# ──────────────────────────────────────────────


def initialize_parameters(
    n_in: int,
    n_h1: int,
    n_h2: int,
    rng: np.random.RandomState,
) -> Params:
    """Initialise the weights: He for the ReLU layers, Xavier for the output layer."""
    return {
        "W1": rng.randn(n_in, n_h1) * np.sqrt(2.0 / n_in),
        "b1": np.zeros(n_h1),
        "W2": rng.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1),
        "b2": np.zeros(n_h2),
        "W3": rng.randn(n_h2, 1) * np.sqrt(1.0 / n_h2),
        "b3": np.zeros(1),
    }


# ──────────────────────────────────────────────
# 3. Forward pass (inverted dropout while training)
# ──────────────────────────────────────────────


def forward(
    X: FloatArray,
    params: Params,
    training: bool,
    rng: np.random.RandomState,
    dropout_p: float,
) -> tuple[FloatArray, ForwardCache]:
    """Run the forward pass.

    With ``training=True`` inverted dropout with probability ``dropout_p`` is
    applied to both hidden layers; with ``training=False`` dropout is the
    identity and the network is deterministic.

    Returns
    -------
    A3 : numpy.ndarray
        Output probabilities, shape ``(n, 1)``.
    cache : ForwardCache
        Intermediate values for ``backward``.
    """
    keep = 1.0 - dropout_p

    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    D1: FloatArray | None = None
    if training:
        D1 = (rng.rand(*A1.shape) < keep).astype(A1.dtype)
        A1 = A1 * D1 / keep

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    D2: FloatArray | None = None
    if training:
        D2 = (rng.rand(*A2.shape) < keep).astype(A2.dtype)
        A2 = A2 * D2 / keep

    Z3 = A2 @ params["W3"] + params["b3"]
    A3 = sigmoid(Z3)

    cache: ForwardCache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "D1": D1,
        "Z2": Z2,
        "A2": A2,
        "D2": D2,
        "Z3": Z3,
        "A3": A3,
    }
    return A3, cache


# ──────────────────────────────────────────────
# 4. Loss: binary cross-entropy
# ──────────────────────────────────────────────


def binary_cross_entropy(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Mean binary cross-entropy with predictions clipped away from 0 and 1."""
    eps = 1e-15
    y_pred = np.clip(y_pred.ravel(), eps, 1.0 - eps)
    y_true = y_true.ravel().astype(np.float64)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


# ──────────────────────────────────────────────
# 5. Backward pass
# ──────────────────────────────────────────────


def backward(
    y_true: FloatArray,
    cache: ForwardCache,
    params: Params,
    dropout_p: float,
) -> Params:
    """Backpropagate the mean BCE loss and return ``dW1, db1, ..., dW3, db3``."""
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


def sgd_step(params: Params, grads: Params, lr: float) -> Params:
    """Apply one vanilla gradient-descent update."""
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        params[key] = params[key] - lr * grads[f"d{key}"]
    return params


# ──────────────────────────────────────────────
# 7. Training loop
# ──────────────────────────────────────────────


def train_mlp(
    X_train: FloatArray,
    y_train: FloatArray,
    X_val: FloatArray,
    y_val: FloatArray,
    *,
    hidden_1: int,
    hidden_2: int,
    dropout_p: float,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    random_state: int,
    verbose: bool = True,
) -> tuple[Params, list[EpochRecord]]:
    """Train with mini-batch SGD, per-epoch shuffling and early stopping on val AUC.

    Parameters
    ----------
    X_train, y_train
        Training data.
    X_val, y_val
        Data used for early stopping.
    hidden_1, hidden_2
        Hidden layer sizes.
    dropout_p
        Dropout probability of both hidden layers.
    learning_rate
        SGD step size.
    batch_size
        Mini-batch size.
    max_epochs
        Maximum number of epochs.
    patience
        Epochs without val AUC improvement before stopping.
    random_state
        Seed of the generator used for init, dropout and shuffling.
    verbose
        Log per-epoch progress (disabled during tuning).

    Returns
    -------
    best_params : dict
        Weights of the epoch with the best validation AUC.
    history : list of dict
        Loss and AUC per epoch.
    """
    rng = np.random.RandomState(random_state)
    n_in = X_train.shape[1]
    n = X_train.shape[0]

    params = initialize_parameters(n_in, hidden_1, hidden_2, rng)

    best_auc = -np.inf
    best_params = {k: v.copy() for k, v in params.items()}
    best_epoch = 0
    epochs_no_improve = 0
    history: list[EpochRecord] = []

    for epoch in range(1, max_epochs + 1):
        perm = rng.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_losses = []
        for start in range(0, n, batch_size):
            Xb = X_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]

            A3, cache = forward(Xb, params, training=True, rng=rng, dropout_p=dropout_p)
            loss = binary_cross_entropy(yb, A3)
            grads = backward(yb, cache, params, dropout_p=dropout_p)
            params = sgd_step(params, grads, learning_rate)
            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses))
        val_proba = predict_proba(X_val, params)
        val_loss = binary_cross_entropy(y_val, val_proba)
        val_auc = float(roc_auc_score(y_val, val_proba))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
            }
        )

        improved = val_auc > best_auc
        if improved:
            best_auc = val_auc
            best_params = {k: v.copy() for k, v in params.items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or improved):
            logger.info(
                "[MLP] epoch %3d | train_loss %.4f | val_loss %.4f | val_auc %.4f%s",
                epoch,
                train_loss,
                val_loss,
                val_auc,
                " *" if improved else "",
            )

        if epochs_no_improve >= patience:
            if verbose:
                logger.info(
                    "[MLP] Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch,
                    patience,
                )
            break

    if verbose:
        logger.info("[MLP] Best val AUC: %.4f (epoch %d)", best_auc, best_epoch)
    return best_params, history


# ──────────────────────────────────────────────
# 8. Prediction
# ──────────────────────────────────────────────


def predict_proba(X: FloatArray, params: Params) -> FloatArray:
    """Return ``P(y=1)`` as a 1-D array (forward pass with dropout disabled)."""
    dummy_rng = np.random.RandomState(0)
    A3, _ = forward(X, params, training=False, rng=dummy_rng, dropout_p=0.0)
    return A3.ravel()


def predict(X: FloatArray, params: Params, threshold: float = 0.5) -> IntArray:
    """Return hard 0/1 predictions."""
    return (predict_proba(X, params) >= threshold).astype(int)


# ──────────────────────────────────────────────
# 9. Hyperparameter tuning (manual search over TimeSeriesSplit)
# ──────────────────────────────────────────────


def tune_hyperparameters(
    X_train: FloatArray,
    y_train: FloatArray,
    cv: TimeSeriesSplit,
    config: ModelSettings,
    scoring: str,
    seed: int,
) -> dict[str, Any]:
    """Manual grid search over ``hidden_pair x learning_rate`` with TimeSeriesSplit.

    Every combination is trained on each fold and scored by the mean fold
    AUC. ``GridSearchCV`` is not used because the MLP is not an sklearn
    estimator.

    Returns
    -------
    dict
        Full hyperparameters: ``config.params`` + ``random_state`` + best grid values.

    Raises
    ------
    ValueError
        If ``scoring`` is not ``"roc_auc"``, the only supported metric.
    """
    if scoring != "roc_auc":
        raise ValueError(f"{MODEL_NAME} tuning only supports scoring='roc_auc', got {scoring!r}")

    default_params = {**config.params, "random_state": seed}
    base_kwargs = {k: v for k, v in default_params.items() if k in TRAIN_KWARGS}
    hidden_pairs = config.grid["hidden_pair"]
    learning_rates = config.grid["learning_rate"]
    results: list[GridResult] = []

    logger.info(
        "[%s] Manual tuning: %d combos x %d folds",
        MODEL_NAME,
        len(hidden_pairs) * len(learning_rates),
        cv.get_n_splits(),
    )

    for (h1, h2), lr in product(hidden_pairs, learning_rates):
        fold_aucs = []
        for tr_idx, va_idx in cv.split(X_train):
            X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
            X_va_f, y_va_f = X_train[va_idx], y_train[va_idx]

            best_params, _ = train_mlp(
                X_tr_f,
                y_tr_f,
                X_va_f,
                y_va_f,
                **{**base_kwargs, "hidden_1": h1, "hidden_2": h2, "learning_rate": lr},
                verbose=False,
            )
            y_proba = predict_proba(X_va_f, best_params)
            fold_aucs.append(float(roc_auc_score(y_va_f, y_proba)))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        results.append(
            {
                "params": {"hidden_1": h1, "hidden_2": h2, "learning_rate": lr},
                "mean_cv_auc": mean_auc,
                "std_cv_auc": std_auc,
            }
        )
        logger.info(
            "[%s]   (H1=%s, H2=%s, lr=%s) -> CV AUC = %.4f ± %.4f",
            MODEL_NAME,
            h1,
            h2,
            lr,
            mean_auc,
            std_auc,
        )

    log_grid_results(MODEL_NAME, results, top_k=5)
    best = max(results, key=lambda r: r["mean_cv_auc"])
    return {**default_params, **best["params"]}


# ──────────────────────────────────────────────
# 10. Public wrapper (contract shared by all models)
# ──────────────────────────────────────────────


def train_and_evaluate(
    X_train: FloatArray,
    y_train: FloatArray,
    X_val: FloatArray,
    y_val: FloatArray,
    feature_cols: list[str],
    params: dict[str, Any],
    plots_dir: Path,
) -> ModelResult:
    """Train the MLP, evaluate it on train and validation and save its plots.

    ``feature_cols`` is unused; it keeps the signature shared by all models.

    Returns
    -------
    dict
        Same keys as the sklearn models; ``model`` is
        ``{"params": weights, "history": history, "config": params}``.
    """
    logger.info("[%s] Training with params: %s", MODEL_NAME, params)

    train_kwargs = {k: v for k, v in params.items() if k in TRAIN_KWARGS}
    best_params, history = train_mlp(X_train, y_train, X_val, y_val, **train_kwargs)

    y_proba_train = predict_proba(X_train, best_params)
    y_proba_val = predict_proba(X_val, best_params)
    y_pred_train = (y_proba_train >= 0.5).astype(int)
    y_pred_val = (y_proba_val >= 0.5).astype(int)

    metrics_train = compute_metrics(y_train, y_pred_train, y_proba_train)
    metrics_val = compute_metrics(y_val, y_pred_val, y_proba_val)

    log_full_metrics(metrics_train, "Train", MODEL_NAME)
    log_full_metrics(metrics_val, "Val", MODEL_NAME)
    logger.info(
        "[%s] Epochs trained: %d | best val AUC: %.4f",
        MODEL_NAME,
        len(history),
        max(h["val_auc"] for h in history),
    )

    plot_confusion_matrix(y_val, y_pred_val, MODEL_NAME, plots_dir)
    plot_roc_curve(y_val, y_proba_val, MODEL_NAME, plots_dir)
    _plot_training_curves(history, plots_dir)

    return {
        "model_name": MODEL_NAME,
        "model": {"params": best_params, "history": history, "config": params},
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
        "y_pred_val": y_pred_val,
        "y_proba_val": y_proba_val,
    }


def _plot_training_curves(history: list[EpochRecord], save_dir: Path) -> Path:
    """Save train/val loss and val AUC per epoch."""
    epochs = [h["epoch"] for h in history]
    train_l = [h["train_loss"] for h in history]
    val_l = [h["val_loss"] for h in history]
    val_aucs = [h["val_auc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_l, label="train_loss")
    axes[0].plot(epochs, val_l, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Loss curve — MLP NumPy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_aucs, color="C2", label="val_auc")
    axes[1].axhline(0.5, linestyle="--", color="gray", label="Random")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val AUC")
    axes[1].set_title("Validation AUC per epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return save_figure(fig, save_dir, "training_curves_mlp_numpy.png")
