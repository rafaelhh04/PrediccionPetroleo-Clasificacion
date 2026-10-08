"""Tests for brent_forecast.models: tuning helpers, sklearn wrappers and the NumPy MLP."""

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.estimator_checks import parametrize_with_checks

from brent_forecast.config import Settings
from brent_forecast.models import neural_network
from brent_forecast.models.base import fit_and_evaluate, log_top_features
from brent_forecast.models.mlp import NumpyMLPClassifier
from brent_forecast.models.registry import (
    MODEL_NAMES,
    build_estimator,
    build_pipeline,
    model_params,
    param_grid,
)
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


def _frames(
    xy: tuple[np.ndarray, np.ndarray],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X, y = xy
    names = ["lag_ret_1", "b", "c", "d", "e"]
    X_df = pd.DataFrame(X, columns=names)
    y_s = pd.Series(y, name="label")
    return X_df[:220], y_s[:220], X_df[220:], y_s[220:]


# ── registry ──────────────────────────────────────────────────


def test_build_estimator_types() -> None:
    assert isinstance(build_estimator("logistic_regression", {}), LogisticRegression)
    assert isinstance(build_estimator("mlp", {"hidden_1": 4}), NumpyMLPClassifier)
    calibrated = build_estimator("svm", {"C": 2.0})
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert calibrated.method == "sigmoid"
    assert calibrated.estimator.C == 2.0
    assert isinstance(build_estimator("svm", {}, for_search=True), SVC)


def test_build_estimator_rejects_unknown_models() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        build_estimator("xgboost", {})


def test_build_pipeline_steps(settings: Settings) -> None:
    pipe = build_pipeline("random_forest", {"n_estimators": 5}, settings.preprocessing)

    assert [name for name, _ in pipe.steps] == ["vif", "winsor", "scale", "model"]
    assert pipe.named_steps["vif"].threshold == settings.preprocessing.vif_threshold
    assert pipe.named_steps["winsor"].lower == settings.preprocessing.winsor_lower
    assert pipe.named_steps["model"].n_estimators == 5


def test_param_grid_prefixes_the_model_step() -> None:
    assert param_grid("svm", {"C": [1, 10], "gamma": ["scale"]}) == [
        {"model__C": [1, 10], "model__gamma": ["scale"]}
    ]


def test_param_grid_expands_mlp_hidden_pairs() -> None:
    grid = param_grid("mlp", {"hidden_pair": [[8, 4], [16, 8]], "learning_rate": [0.1]})

    assert grid == [
        {"model__hidden_1": [8], "model__hidden_2": [4], "model__learning_rate": [0.1]},
        {"model__hidden_1": [16], "model__hidden_2": [8], "model__learning_rate": [0.1]},
    ]
    assert model_params({"model__C": 1.0, "model__gamma": "scale"}) == {"C": 1.0, "gamma": "scale"}


@pytest.mark.parametrize("key", list(MODEL_NAMES))
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_every_model_pipeline_tunes_fits_and_serialises(
    key: str,
    fast_settings: Settings,
    toy_xy: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    X_tr, y_tr, X_va, y_va = _frames(toy_xy)
    config = getattr(fast_settings.models, key)
    fixed = {**config.params, "random_state": 7}

    best = grid_search(
        build_pipeline(key, fixed, fast_settings.preprocessing, for_search=True),
        param_grid(key, config.grid),
        X_tr,
        y_tr,
        cv=make_time_series_cv(2),
        scoring="roc_auc",
        n_jobs=1,
        model_name=MODEL_NAMES[key],
    )
    pipe = build_pipeline(key, {**fixed, **model_params(best)}, fast_settings.preprocessing)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_va)[:, 1]

    assert proba.shape == (len(X_va),)
    assert ((proba >= 0) & (proba <= 1)).all()
    path = tmp_path / f"{key}.joblib"
    joblib.dump(pipe, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(X_va)[:, 1], proba)


def test_log_top_features(
    toy_xy: tuple[np.ndarray, np.ndarray], settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    X_tr, y_tr, _, _ = _frames(toy_xy)

    for key in ("logistic_regression", "random_forest", "mlp"):
        params = {"n_estimators": 10, "random_state": 0} if key == "random_forest" else {}
        if key == "mlp":
            params = {"max_epochs": 2, "random_state": 0}
        pipe = build_pipeline(key, params, settings.preprocessing).fit(X_tr, y_tr)
        log_top_features(pipe, key)

    assert "+ lag_ret_1" in caplog.text  # informative feature, positive coefficient
    assert "Top 5 features by importance" in caplog.text
    assert "[mlp]" not in caplog.text  # no coefficients/importances to report


# ── NumPy MLP as a scikit-learn estimator ─────────────────────


@parametrize_with_checks([NumpyMLPClassifier(random_state=0)])
def test_mlp_is_a_valid_sklearn_classifier(estimator, check) -> None:
    check(estimator)


def test_mlp_fit_predict_and_attributes(toy_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = toy_xy
    labels = np.where(y == 1, "up", "down")

    mlp = NumpyMLPClassifier(
        hidden_1=8, hidden_2=4, learning_rate=0.1, max_epochs=30, random_state=0
    ).fit(X, labels)

    assert mlp.classes_.tolist() == ["down", "up"]
    assert set(mlp.predict(X)) <= {"down", "up"}
    assert mlp.predict_proba(X).shape == (len(X), 2)
    np.testing.assert_allclose(mlp.predict_proba(X).sum(axis=1), 1.0)
    assert 1 <= mlp.n_epochs_ == len(mlp.history_) <= 30
    assert (mlp.predict(X) == labels).mean() > 0.8


def test_mlp_rejects_multiclass_and_bad_validation_fraction() -> None:
    X = np.random.default_rng(0).normal(size=(30, 2))

    with pytest.raises(ValueError, match="Only binary classification"):
        NumpyMLPClassifier().fit(X, np.arange(30) % 3)
    with pytest.raises(ValueError, match="validation_fraction"):
        NumpyMLPClassifier(validation_fraction=1.0).fit(X, np.arange(30) % 2)


def test_mlp_without_validation_keeps_the_last_epoch(toy_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = toy_xy

    mlp = NumpyMLPClassifier(max_epochs=7, patience=1, validation_fraction=0.0, random_state=0)
    mlp.fit(X, y)

    assert mlp.n_epochs_ == 7  # no early stopping without a hold-out


def test_mlp_holds_out_the_last_rows_for_early_stopping(
    toy_xy: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y = toy_xy
    seen = {}

    def fake_train(X_tr, y_tr, X_val, y_val, **kwargs):
        seen.update(n_train=len(X_tr), n_val=len(X_val), X_val=X_val, monitor=kwargs["monitor"])
        return neural_network.train_mlp(X_tr, y_tr, X_val, y_val, **kwargs)

    monkeypatch.setattr("brent_forecast.models.mlp.train_mlp", fake_train)

    NumpyMLPClassifier(max_epochs=2, validation_fraction=0.2, random_state=0).fit(X, y)

    assert (seen["n_train"], seen["n_val"]) == (240, 60)
    np.testing.assert_array_equal(seen["X_val"], X[-60:])  # chronological, not shuffled
    assert seen["monitor"] == "loss"


def test_mlp_is_picklable_and_deterministic(toy_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = toy_xy
    a = NumpyMLPClassifier(max_epochs=5, random_state=3).fit(X, y)
    b = NumpyMLPClassifier(max_epochs=5, random_state=3).fit(X, y)

    restored = pickle.loads(pickle.dumps(a))

    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    np.testing.assert_array_equal(restored.predict_proba(X), a.predict_proba(X))


def test_mlp_in_a_pipeline_receives_dataframes(
    toy_xy: tuple[np.ndarray, np.ndarray], settings: Settings
) -> None:
    X_tr, y_tr, X_va, _ = _frames(toy_xy)

    pipe = build_pipeline("mlp", {"max_epochs": 3, "random_state": 0}, settings.preprocessing)

    assert isinstance(pipe, Pipeline)
    assert pipe.fit(X_tr, y_tr).predict_proba(X_va).shape == (len(X_va), 2)


# ── NumPy core (behaviour; gradients are checked in test_mlp_gradients) ──


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
    assert "Best monitored auc" in caplog.text


def test_predict_returns_hard_labels(toy_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, _ = toy_xy
    params = neural_network.initialize_parameters(5, 4, 2, np.random.RandomState(0))

    proba = neural_network.predict_proba(X, params)
    hard = neural_network.predict(X, params, threshold=0.5)

    assert proba.shape == (len(X),)
    np.testing.assert_array_equal(hard, (proba >= 0.5).astype(int))
