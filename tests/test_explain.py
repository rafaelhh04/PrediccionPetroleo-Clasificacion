"""Tests for brent_forecast.evaluation.explain (``brent explain``)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from brent_forecast.config import Settings
from brent_forecast.evaluation.explain import (
    generate_explanations,
    permutation_importances,
    plot_permutation,
    plot_shap,
    shap_importances,
    shap_values,
)
from brent_forecast.models.registry import build_pipeline


def _data(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Signal only in ``signal``; ``copy`` duplicates it so that VIF drops one of the two."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["signal", "noise", "lag_ret_1", "vix"])
    X["copy"] = X["signal"] + rng.normal(0, 1e-3, n)
    y = pd.Series((2 * X["signal"] + rng.normal(size=n) > 0).astype(float))
    return X, y


def _fitted(key: str, settings: Settings, params: dict) -> tuple:
    X, y = _data()
    pipe = build_pipeline(key, {"random_state": 0, **params}, settings.preprocessing)
    return pipe.fit(X.iloc[:400], y.iloc[:400]), X.iloc[400:], y.iloc[400:], X.iloc[:400]


def test_permutation_importance_finds_the_signal_and_ignores_dropped_columns(
    settings: Settings,
) -> None:
    pipe, X_test, y_test, _ = _fitted("logistic_regression", settings, {})

    table = permutation_importances(pipe, X_test, y_test, n_repeats=5, seed=0)

    kept = set(pipe.named_steps["vif"].get_feature_names_out())
    dropped = ({"signal", "copy"} - kept).pop()  # VIF keeps one of the collinear pair
    assert table["feature"].iloc[0] == ({"signal", "copy"} & kept).pop()
    assert table["importance_mean"].iloc[0] > 0.2
    assert table.set_index("feature").loc[dropped, "importance_mean"] == 0.0
    assert list(table.columns) == ["feature", "importance_mean", "importance_std"]


@pytest.mark.parametrize(
    ("key", "params", "output"),
    [
        ("logistic_regression", {}, "log-odds"),
        ("lightgbm", {"n_estimators": 20, "verbose": -1}, "log-odds"),
        ("random_forest", {"n_estimators": 20, "max_depth": 3}, "probability"),
        ("mlp", {"hidden_1": 4, "hidden_2": 2, "max_epochs": 3}, "probability"),
    ],
)
def test_shap_values_are_additive_and_rank_the_signal_first(
    settings: Settings, key: str, params: dict, output: str
) -> None:
    pipe, X_test, _, background = _fitted(key, settings, params)
    explained = X_test.iloc[:40]

    result = shap_values(pipe, background.iloc[:50], explained, seed=0)

    model, inputs = pipe[-1], pipe[:-1].transform(explained)
    reconstructed = result.explanation.base_values + result.explanation.values.sum(axis=1)
    if output == "log-odds":
        proba = model.predict_proba(inputs)[:, 1]
        expected = np.log(proba / (1 - proba))
    else:
        expected = model.predict_proba(inputs)[:, 1]
    np.testing.assert_allclose(reconstructed, expected, atol=1e-4)
    assert result.output == output
    assert result.explanation.values.shape == inputs.shape
    if key != "mlp":  # 3 epochs are not enough for the MLP to learn the signal
        assert shap_importances(result)["feature"].iloc[0] in {"signal", "copy"}


def test_explanation_plots(settings: Settings, plots_dir: Path) -> None:
    pipe, X_test, y_test, background = _fitted("logistic_regression", settings, {})
    result = shap_values(pipe, background.iloc[:50], X_test.iloc[:30], seed=0)
    table = permutation_importances(pipe, X_test, y_test, n_repeats=2, seed=0)

    paths = [plot_permutation(table, "Logistic Regression", plots_dir)]
    paths += plot_shap(result, "Logistic Regression", plots_dir)

    assert [p.name for p in paths] == [
        "permutation_importance_logistic_regression.png",
        "shap_beeswarm_logistic_regression.png",
        "shap_waterfall_logistic_regression.png",
    ]
    assert all(p.stat().st_size > 0 for p in paths)


def test_generate_explanations_needs_trained_models(fast_settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="brent train"):
        generate_explanations(fast_settings)


def test_generate_explanations_rejects_unknown_models(
    fast_settings: Settings, tmp_path: Path
) -> None:
    settings = fast_settings.model_copy(
        update={"explain": fast_settings.explain.model_copy(update={"models": ["xgboost"]})}
    )

    with pytest.raises(KeyError, match="xgboost"):
        generate_explanations(settings)


def test_explained_models_are_refitted_on_the_development_period(
    fast_settings: Settings, small_raw_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brent_forecast.pipeline import prepare_data

    settings = fast_settings.model_copy(
        update={
            "explain": fast_settings.explain.model_copy(
                update={"models": ["logistic_regression"], "permutation_repeats": 2}
            )
        }
    )
    paths = settings.paths
    paths.models_dir.mkdir(parents=True)
    joblib.dump(
        build_pipeline("logistic_regression", {"C": 0.5}, settings.preprocessing),
        paths.models_dir / "logistic_regression.joblib",
    )
    seen: list[pd.Index] = []
    original_fit = type(build_pipeline("logistic_regression", {}, settings.preprocessing)).fit

    def spy_fit(self, X, y=None, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(X.index)
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr("sklearn.pipeline.Pipeline.fit", spy_fit)

    summary = generate_explanations(settings)

    _, parts = prepare_data(settings)
    assert len(seen) == 1
    assert seen[0].equals(parts.dev.index)  # never the out-of-sample rows
    text = summary.read_text()
    assert "## Logistic Regression" in text
    assert "multiple comparisons" in text
    assert (paths.explain_dir / "permutation_logistic_regression.csv").is_file()
    assert (paths.explain_dir / "shap_logistic_regression.csv").is_file()
