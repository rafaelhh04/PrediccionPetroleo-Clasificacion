"""Tests for brent_forecast.models: tuning helpers, sklearn wrappers and the NumPy MLP."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from brent_forecast.config import ModelSettings, Settings
from brent_forecast.models import logistic_regression, neural_network, random_forest, svm
from brent_forecast.models.base import fit_and_evaluate
from brent_forecast.models.tuning import grid_search, log_grid_results, make_time_series_cv

RESULT_KEYS = {"model_name", "model", "metrics_train", "metrics_val", "y_pred_val", "y_proba_val"}
FEATURES = [f"f{i}" for i in range(5)]


def _split(xy: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, ...]:
    X, y = xy
    return X[:220], y[:220], X[220:], y[220:]


# ── tuning helpers ────────────────────────────────────────────


def test_make_time_series_cv_is_expanding_and_ordered() -> None:
    cv = make_time_series_cv(3)
    folds = list(cv.split(np.zeros((40, 1))))

    assert isinstance(cv, TimeSeriesSplit)
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        assert train_idx.max() < val_idx.min()
    assert [len(tr) for tr, _ in folds] == sorted(len(tr) for tr, _ in folds)


def test_grid_search_returns_best_params_and_logs(
    toy_xy: tuple[np.ndarray, np.ndarray], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    X, y = toy_xy

    best = grid_search(
        LogisticRegression(),
        {"C": [1e-6, 1.0]},
        X,
        y,
        cv=make_time_series_cv(3),
        scoring="roc_auc",
        n_jobs=1,
        model_name="LR",
    )

    assert set(best) == {"C"}
    assert best["C"] in (1e-6, 1.0)
    assert "[LR] Top 2 configurations" in caplog.text


def test_log_grid_results_sorts_and_truncates(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    results = [
        {"params": {"k": i}, "mean_cv_auc": auc, "std_cv_auc": 0.01}
        for i, auc in enumerate([0.51, 0.58, 0.49, 0.55])
    ]

    log_grid_results("M", results, top_k=2)

    assert "Top 2 configurations" in caplog.text
    assert "Best params: {'k': 1}" in caplog.text
    assert "0.4900" not in caplog.text


# ── shared sklearn evaluation ─────────────────────────────────


def test_fit_and_evaluate(toy_xy: tuple[np.ndarray, np.ndarray], plots_dir: Path) -> None:
    X_tr, y_tr, X_va, y_va = _split(toy_xy)
    model = LogisticRegression()

    result = fit_and_evaluate(model, "My Model", X_tr, y_tr, X_va, y_va, plots_dir)

    assert set(result) == RESULT_KEYS
    assert result["model"] is model
    assert result["y_proba_val"].shape == (len(y_va),)
    assert result["metrics_val"]["auc_roc"] > 0.8
    assert (plots_dir / "confusion_my_model.png").is_file()
    assert (plots_dir / "roc_my_model.png").is_file()


# ── sklearn model modules ─────────────────────────────────────

SKLEARN_CASES = [
    (logistic_regression, "logistic_regression", "Logistic Regression"),
    (svm, "svm", "SVM (RBF)"),
    (random_forest, "random_forest", "Random Forest"),
]


@pytest.mark.parametrize(("module", "key", "name"), SKLEARN_CASES)
def test_sklearn_tune_then_train(
    module: Any,
    key: str,
    name: str,
    fast_settings: Settings,
    toy_xy: tuple[np.ndarray, np.ndarray],
    plots_dir: Path,
) -> None:
    X_tr, y_tr, X_va, y_va = _split(toy_xy)
    config: ModelSettings = getattr(fast_settings.models, key)

    params = module.tune_hyperparameters(
        X_tr, y_tr, make_time_series_cv(2), config, scoring="roc_auc", seed=7
    )

    best = {k: params[k] for k in config.grid}
    assert params == {**config.params, "random_state": 7, **best}
    assert all(best[k] in values for k, values in config.grid.items())

    result = module.train_and_evaluate(X_tr, y_tr, X_va, y_va, FEATURES, params, plots_dir)

    assert result["model_name"] == name
    assert set(result) == RESULT_KEYS
    assert result["metrics_val"]["auc_roc"] > 0.7


def test_svm_search_disables_probability_but_final_model_keeps_it(
    fast_settings: Settings, toy_xy: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y = toy_xy
    seen: list[dict[str, Any]] = []

    def spy(estimator: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(estimator.get_params())
        return {"C": 1.0}

    monkeypatch.setattr(svm, "grid_search", spy)

    params = svm.tune_hyperparameters(
        X, y, make_time_series_cv(2), fast_settings.models.svm, scoring="roc_auc", seed=0
    )

    assert seen[0]["probability"] is False
    assert params["probability"] is True


def test_logistic_regression_logs_top_coefficients(
    toy_xy: tuple[np.ndarray, np.ndarray], plots_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    X_tr, y_tr, X_va, y_va = _split(toy_xy)

    logistic_regression.train_and_evaluate(X_tr, y_tr, X_va, y_va, FEATURES, {}, plots_dir)

    assert "Top 5 features by |coef|" in caplog.text
    assert "+ f0" in caplog.text  # the informative feature has a positive coefficient


# ── NumPy MLP (behaviour; gradients are checked in test_mlp_gradients) ──


def test_mlp_tune_then_train(
    fast_settings: Settings, toy_xy: tuple[np.ndarray, np.ndarray], plots_dir: Path
) -> None:
    X_tr, y_tr, X_va, y_va = _split(toy_xy)
    config = fast_settings.models.mlp

    params = neural_network.tune_hyperparameters(
        X_tr, y_tr, make_time_series_cv(2), config, scoring="roc_auc", seed=3
    )

    assert params == {**config.params, "random_state": 3, "hidden_1": 8, "hidden_2": 4, "learning_rate": 0.01}

    result = neural_network.train_and_evaluate(X_tr, y_tr, X_va, y_va, FEATURES, params, plots_dir)

    assert set(result) == RESULT_KEYS
    assert set(result["model"]) == {"params", "history", "config"}
    assert 1 <= len(result["model"]["history"]) <= config.params["max_epochs"]
    assert (plots_dir / "training_curves_mlp_numpy.png").is_file()


def test_mlp_tuning_rejects_other_scorings(
    fast_settings: Settings, toy_xy: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = toy_xy

    with pytest.raises(ValueError, match="roc_auc"):
        neural_network.tune_hyperparameters(
            X, y, make_time_series_cv(2), fast_settings.models.mlp, scoring="accuracy", seed=0
        )


def test_train_mlp_logs_progress_and_early_stops(
    toy_xy: tuple[np.ndarray, np.ndarray], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    X_tr, y_tr, X_va, y_va = _split(toy_xy)

    _, history = neural_network.train_mlp(
        X_tr,
        y_tr,
        X_va,
        y_va,
        hidden_1=4,
        hidden_2=2,
        dropout_p=0.0,
        learning_rate=0.0,  # no learning -> no improvement after epoch 1
        batch_size=32,
        max_epochs=50,
        patience=3,
        random_state=0,
    )

    assert len(history) == 4
    assert "Early stopping at epoch 4" in caplog.text
    assert "Best val AUC" in caplog.text


def test_predict_returns_hard_labels(toy_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = toy_xy
    params = neural_network.initialize_parameters(5, 4, 2, np.random.RandomState(0))

    proba = neural_network.predict_proba(X, params)
    hard = neural_network.predict(X, params, threshold=0.5)

    assert proba.shape == (len(X),)
    np.testing.assert_array_equal(hard, (proba >= 0.5).astype(int))
