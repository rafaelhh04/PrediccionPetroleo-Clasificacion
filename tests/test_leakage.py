"""Leakage guards: tests that fail if information from the future reaches the past.

Three families:

1. Everything fitted during preprocessing (VIF selection, winsorisation
   percentiles, scaler) depends on training rows only.
2. Future permutation: rewriting every row after a cut-off never changes the
   features of rows up to the cut-off, and a label only depends on the next day.
3. The pipeline tunes on the development period only and evaluates the
   out-of-sample period with purged walk-forward windows.

The purge removes the last training day before every test window: its label is
the direction of the first test day. Thanks to it, corrupting the future does
not change a single training label either (no boundary exception needed).
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from brent_forecast import pipeline
from brent_forecast.config import PreprocessingSettings, Settings, SplitSettings
from brent_forecast.features.preprocessing import (
    build_dataset,
    create_label,
    engineer_features,
    handle_nulls,
    split_by_date,
)
from brent_forecast.features.transformers import VIFSelector, Winsorizer
from brent_forecast.models.registry import build_pipeline
from brent_forecast.validation.walk_forward import PurgedWalkForwardSplit

SPLIT = SplitSettings(test_start="2024-01-01")
PURGE, EMBARGO = 1, 5
PREP = PreprocessingSettings(vif_threshold=10.0, winsor_lower=0.01, winsor_upper=0.99)
TEST_START = pd.Timestamp("2024-01-01")
NUMERIC_RAW = [
    "brent_price",
    "wti_price",
    "dxy_index",
    "vix",
    "gpr_index",
    "brent_return",
    "wti_return",
    "brent_lag_1",
    "brent_lag_3",
    "brent_lag_7",
    "wti_lag_1",
    "wti_lag_3",
    "wti_lag_7",
    "brent_volatility_7d",
    "brent_volatility_30d",
    "wti_volatility_7d",
    "wti_volatility_30d",
    "brent_wti_spread",
]


def _corrupt_after(df: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Overwrite every numeric/event value strictly after ``cutoff`` with extreme values."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    future = out["date"] > cutoff
    n = int(future.sum())
    for col in NUMERIC_RAW:
        # Positive so that log returns stay defined and the corrupted rows are kept.
        out.loc[future, col] = rng.uniform(1e3, 1e6, n)
    out.loc[future, "brent_return"] *= rng.choice([-1, 1], n)
    out.loc[future, "event_severity"] = rng.integers(0, 11, n)
    out.loc[future, "event_type"] = "war"
    return out


# ── 1. fitted preprocessing state depends on train only ───────


def _first_window_training_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Training rows of the first walk-forward window over the out-of-sample period."""
    dataset = build_dataset(frame)
    n_dev = len(split_by_date(dataset.frame, SPLIT).dev)
    cv = PurgedWalkForwardSplit(
        None, test_size=63, first_test_index=n_dev, purge=PURGE, embargo=EMBARGO
    )
    train_idx, _ = next(cv.split(dataset.features))
    return dataset.features.iloc[train_idx], dataset.target.iloc[train_idx]


def test_first_window_training_data_ignores_out_of_sample_values(
    merged_frame: pd.DataFrame,
) -> None:
    last_dev_day = merged_frame.loc[merged_frame["date"] < TEST_START, "date"].max()
    corrupted = _corrupt_after(merged_frame, last_dev_day)  # returns and labels included

    X_clean, y_clean = _first_window_training_rows(merged_frame.copy())
    X_dirty, y_dirty = _first_window_training_rows(corrupted)

    pd.testing.assert_frame_equal(X_dirty, X_clean)
    pd.testing.assert_series_equal(y_dirty, y_clean)  # every label, thanks to the purge

    prep, prep_d = (
        build_pipeline("logistic_regression", {}, PREP)[:-1].fit(X) for X in (X_clean, X_dirty)
    )
    vif, vif_d = prep.named_steps["vif"], prep_d.named_steps["vif"]
    np.testing.assert_array_equal(vif_d.support_, vif.support_)  # VIF selection
    win, win_d = prep.named_steps["winsor"], prep_d.named_steps["winsor"]
    np.testing.assert_array_equal(win_d.lower_bounds_, win.lower_bounds_)  # winsorisation
    np.testing.assert_array_equal(win_d.upper_bounds_, win.upper_bounds_)
    sc, sc_d = prep.named_steps["scale"], prep_d.named_steps["scale"]
    np.testing.assert_array_equal(sc_d.mean_, sc.mean_)  # scaler
    np.testing.assert_array_equal(sc_d.scale_, sc.scale_)


def test_without_purge_the_last_training_label_would_leak(merged_frame: pd.DataFrame) -> None:
    """Control: the label of the last development day is realised on the first test day."""
    flipped = merged_frame.copy()
    first_test_row = flipped.index[flipped["date"] >= TEST_START][0]
    flipped.loc[first_test_row, "brent_return"] *= -1

    dev_clean = split_by_date(build_dataset(merged_frame.copy()).frame, SPLIT).dev
    dev_flipped = split_by_date(build_dataset(flipped).frame, SPLIT).dev

    changed = dev_clean["label"] != dev_flipped["label"]
    assert changed.tolist() == [False] * (len(changed) - 1) + [True]


def test_vif_selection_ignores_non_training_rows() -> None:
    rng = np.random.default_rng(1)
    n_train, n_future = 300, 3000
    X = pd.DataFrame({c: rng.normal(size=n_train + n_future) for c in "abcd"})
    X_collinear = X.copy()
    X_collinear.loc[n_train:, "b"] = X.loc[n_train:, "a"] * 2  # collinear only in the future

    kept = VIFSelector(5.0).fit(X.iloc[:n_train]).get_feature_names_out()
    kept_collinear = VIFSelector(5.0).fit(X_collinear.iloc[:n_train]).get_feature_names_out()
    leaky = VIFSelector(5.0).fit(X_collinear).get_feature_names_out()

    assert kept.tolist() == kept_collinear.tolist() == list("abcd")
    assert len(leaky) == 3  # sanity check: fitted on every row, the selection would change


def test_winsorisation_bounds_come_from_train_only() -> None:
    rng = np.random.default_rng(2)
    X_train = pd.DataFrame(rng.normal(size=(500, 2)), columns=["ret", "ret_2"])
    calm = pd.DataFrame(rng.normal(size=(50, 2)), columns=["ret", "ret_2"])
    wild = calm * 1e6

    winsor = Winsorizer(0.01, 0.99, name_contains="ret").fit(X_train)
    out = winsor.transform(wild)

    np.testing.assert_array_equal(winsor.lower_bounds_, np.percentile(X_train, 1, axis=0))
    np.testing.assert_array_equal(winsor.upper_bounds_, np.percentile(X_train, 99, axis=0))
    assert (out <= winsor.upper_bounds_).all()
    assert (out >= winsor.lower_bounds_).all()


# ── 2. future permutation ─────────────────────────────────────


def _features_up_to(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    labelled = create_label(df.copy())
    engineered, cols = engineer_features(labelled)
    cleaned = handle_nulls(engineered, cols)
    return cleaned.loc[cleaned["date"] <= cutoff, ["date", *cols]].reset_index(drop=True)


@pytest.mark.parametrize("position", [0.25, 0.5, 0.8, 0.95])
def test_rewriting_the_future_never_changes_past_features(
    merged_frame: pd.DataFrame, position: float
) -> None:
    cutoff = merged_frame["date"].iloc[int(len(merged_frame) * position)]

    original = _features_up_to(merged_frame, cutoff)
    rewritten = _features_up_to(
        _corrupt_after(merged_frame, cutoff, seed=int(position * 100)), cutoff
    )

    pd.testing.assert_frame_equal(original, rewritten)


def test_label_depends_only_on_the_next_day(merged_frame: pd.DataFrame) -> None:
    t = 1500
    base = create_label(merged_frame.copy())

    beyond = merged_frame.copy()
    beyond.loc[t + 2 :, "brent_return"] = -beyond.loc[t + 2 :, "brent_return"]
    flipped_next = merged_frame.copy()
    flipped_next.loc[t + 1, "brent_return"] = -flipped_next.loc[t + 1, "brent_return"]

    # Rows after t+1 never influence labels up to t ...
    pd.testing.assert_series_equal(
        create_label(beyond)["label"].iloc[: t + 1], base["label"].iloc[: t + 1]
    )
    # ... while the return of t+1 decides label t, and only label t.
    changed = create_label(flipped_next)["label"] != base["label"]
    assert changed[changed].index.tolist() == [t]


def test_no_same_day_or_target_columns_in_features(merged_frame: pd.DataFrame) -> None:
    _, cols = engineer_features(create_label(merged_frame.copy()))

    forbidden = {"label", "brent_return", "wti_return", "brent_price", "wti_price", "date"}
    assert forbidden.isdisjoint(cols)


# ── 3. the pipeline never feeds the test set to tuning/training ──


def test_pipeline_tunes_on_development_rows_and_evaluates_out_of_sample_by_walk_forward(
    fast_settings: Settings, small_raw_data_dir: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, list[tuple[Any, ...]]] = {"tune": [], "curves": [], "walk": []}
    splits: dict[str, Any] = {}

    def spy_split(*args: Any, **kwargs: Any) -> Any:
        parts = split_by_date(*args, **kwargs)
        splits.update(dev=parts.dev.index, test=parts.test.index)
        return parts

    def spy(kind: str, fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            seen[kind].append(args)
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(pipeline, "split_by_date", spy_split)
    monkeypatch.setattr(pipeline, "grid_search", spy("tune", pipeline.grid_search))
    monkeypatch.setattr(pipeline, "plot_learning_curve", spy("curves", lambda *a, **k: None))
    monkeypatch.setattr(
        pipeline, "walk_forward_predict", spy("walk", pipeline.walk_forward_predict)
    )

    pipeline.run(fast_settings)

    dev, test = splits["dev"], splits["test"]
    assert len(seen["tune"]) == len(seen["curves"]) == 4  # models only
    assert len(seen["walk"]) == 8  # models and baselines share the walk-forward
    for args in seen["tune"]:  # grid_search(estimator, grid, X, y, ...)
        assert args[2].index.equals(dev)
        assert args[2].index.intersection(test).empty
    for args in seen["curves"]:  # plot_learning_curve(estimator, X, y, ...)
        assert args[1].index.equals(dev)
    val = fast_settings.validation
    for args in seen["walk"]:  # walk_forward_predict(estimator, X_all, y_all, cv)
        cv = args[3]
        assert cv.first_test_index == len(dev)
        assert (cv.purge, cv.embargo) == (val.purge, val.embargo)
        for train_idx, test_idx in cv.split(args[1]):
            assert train_idx.max() + cv.gap < test_idx.min()

    predictions = pd.read_csv(fast_settings.paths.predictions_file, parse_dates=["date"])
    assert len(predictions) == len(test)
    assert predictions["date"].min() >= TEST_START
