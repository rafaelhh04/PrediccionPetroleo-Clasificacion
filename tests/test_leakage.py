"""Leakage guards: tests that fail if information from the future reaches the past.

Three families:

1. Everything fitted during preprocessing (VIF selection, winsorisation
   percentiles, scaler) depends on training rows only.
2. Future permutation: rewriting every row after a cut-off never changes the
   features of rows up to the cut-off, and a label only depends on the next day.
3. The pipeline never feeds the test set to tuning or training.

Known, accepted boundary effect: the label of the last training day is the
direction of the first validation day (no purge/embargo gap yet; planned for
the walk-forward phase). Tests exclude that single label explicitly.
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
    split_temporal,
)
from brent_forecast.features.transformers import VIFSelector, Winsorizer
from brent_forecast.models.registry import build_pipeline

SPLIT = SplitSettings(train_end="2022-01-01", val_end="2024-01-01")
PREP = PreprocessingSettings(vif_threshold=10.0, winsor_lower=0.01, winsor_upper=0.99)
TRAIN_END = pd.Timestamp("2022-01-01")
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


def _fitted_preprocessing(frame: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Fit the preprocessing part of a model pipeline on the training rows of ``frame``."""
    dataset = build_dataset(frame)
    train = split_temporal(dataset.frame, SPLIT).train
    prep = build_pipeline("logistic_regression", {}, PREP)[:-1]
    X_train = train[dataset.feature_cols]
    return prep.fit(X_train), X_train


def test_fitted_preprocessing_ignores_val_and_test_values(merged_frame: pd.DataFrame) -> None:
    last_train_day = merged_frame.loc[merged_frame["date"] < TRAIN_END, "date"].max()
    corrupted = _corrupt_after(merged_frame, last_train_day)

    clean, X_clean = _fitted_preprocessing(merged_frame.copy())
    dirty, X_dirty = _fitted_preprocessing(corrupted)

    pd.testing.assert_frame_equal(X_dirty, X_clean)  # the training features themselves
    vif, vif_d = clean.named_steps["vif"], dirty.named_steps["vif"]
    np.testing.assert_array_equal(vif_d.support_, vif.support_)  # VIF selection
    win, win_d = clean.named_steps["winsor"], dirty.named_steps["winsor"]
    np.testing.assert_array_equal(win_d.lower_bounds_, win.lower_bounds_)  # winsorisation
    np.testing.assert_array_equal(win_d.upper_bounds_, win.upper_bounds_)
    sc, sc_d = clean.named_steps["scale"], dirty.named_steps["scale"]
    np.testing.assert_array_equal(sc_d.mean_, sc.mean_)  # scaler
    np.testing.assert_array_equal(sc_d.scale_, sc.scale_)
    pd.testing.assert_frame_equal(dirty.transform(X_dirty), clean.transform(X_clean))


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


def test_pipeline_keeps_x_test_out_of_tuning_and_training(
    fast_settings: Settings, small_raw_data_dir: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, list[tuple[Any, ...]]] = {"tune": [], "train": [], "curves": [], "final": []}
    splits: dict[str, Any] = {}

    def spy_split(*args: Any, **kwargs: Any) -> Any:
        parts = split_temporal(*args, **kwargs)
        splits.update(train=parts.train.index, val=parts.val.index, test=parts.test.index)
        return parts

    def spy(kind: str, fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            seen[kind].append(args)
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(pipeline, "split_temporal", spy_split)
    monkeypatch.setattr(pipeline, "grid_search", spy("tune", pipeline.grid_search))
    monkeypatch.setattr(pipeline, "fit_and_evaluate", spy("train", pipeline.fit_and_evaluate))
    monkeypatch.setattr(pipeline, "plot_learning_curve", spy("curves", lambda *a, **k: None))
    monkeypatch.setattr(pipeline, "evaluate_on_test", spy("final", pipeline.evaluate_on_test))

    pipeline.run(fast_settings)

    def rows(kind: str, position: int) -> list[pd.Index]:
        return [args[position].index for args in seen[kind]]

    train, val, test = splits["train"], splits["val"], splits["test"]
    assert len(seen["tune"]) == len(seen["train"]) == len(seen["curves"]) == 4
    assert all(idx.equals(train) for idx in rows("tune", 2))  # grid_search(est, grid, X, y)
    assert all(idx.equals(train) for idx in rows("train", 2))  # fit_and_evaluate(..., X_tr, ...)
    assert all(idx.equals(val) for idx in rows("train", 4))  # ..., X_val, ...)
    assert all(idx.equals(train) for idx in rows("curves", 1))  # plot_learning_curve(est, X, y)
    for kind in ("tune", "train", "curves"):
        for args in seen[kind]:
            frames = [a for a in args if isinstance(a, pd.DataFrame | pd.Series)]
            assert all(f.index.intersection(test).empty for f in frames)
    assert len(seen["final"]) == 1  # the test set is evaluated exactly once
    assert seen["final"][0][1].index.equals(test)
