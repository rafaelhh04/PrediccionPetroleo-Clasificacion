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
    create_label,
    engineer_features,
    filter_features_by_vif,
    handle_nulls,
    preprocess,
    scale_features,
    winsorize_features,
)

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


def test_preprocess_ignores_val_and_test_values(merged_frame: pd.DataFrame) -> None:
    last_train_day = merged_frame.loc[merged_frame["date"] < TRAIN_END, "date"].max()
    corrupted = _corrupt_after(merged_frame, last_train_day)

    clean = preprocess(merged_frame.copy(), SPLIT, PREP)
    dirty = preprocess(corrupted, SPLIT, PREP)

    X_tr, _, _, y_tr, _, _, scaler, cols = clean
    X_tr_d, _, _, y_tr_d, _, _, scaler_d, cols_d = dirty
    assert cols_d == cols  # VIF selection
    np.testing.assert_array_equal(scaler_d.mean_, scaler.mean_)  # scaler fit
    np.testing.assert_array_equal(scaler_d.scale_, scaler.scale_)
    np.testing.assert_array_equal(X_tr_d, X_tr)  # winsorisation + scaling of train
    np.testing.assert_array_equal(y_tr_d[:-1], y_tr[:-1])  # see module docstring


def test_vif_selection_ignores_non_training_rows() -> None:
    rng = np.random.default_rng(1)
    n_train, n_future = 300, 300
    dates = pd.bdate_range("2020-01-01", periods=n_train + n_future)
    df = pd.DataFrame({"date": dates, **{c: rng.normal(size=len(dates)) for c in "abcd"}})
    cutoff = dates[n_train]
    future = df["date"] >= cutoff
    df_collinear = df.copy()
    df_collinear.loc[future, "b"] = df.loc[future, "a"] * 2  # collinear only in the future

    kept = filter_features_by_vif(df, list("abcd"), cutoff, 10.0)
    kept_collinear = filter_features_by_vif(df_collinear, list("abcd"), cutoff, 10.0)

    assert kept == kept_collinear == list("abcd")


def test_winsorisation_bounds_come_from_train_only() -> None:
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(500, 2))
    calm = rng.normal(size=(50, 2))
    wild = calm * 1e6

    train_a, val_a, _ = winsorize_features(
        X_train.copy(), calm.copy(), calm.copy(), ["ret", "ret_2"], 0.01, 0.99
    )
    train_b, val_b, _ = winsorize_features(
        X_train.copy(), wild.copy(), wild.copy(), ["ret", "ret_2"], 0.01, 0.99
    )

    np.testing.assert_array_equal(train_a, train_b)
    hi = np.percentile(X_train, 99, axis=0)
    lo = np.percentile(X_train, 1, axis=0)
    assert (val_b <= hi).all()
    assert (val_b >= lo).all()


def test_scaler_parameters_come_from_train_only() -> None:
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(200, 3))

    *_, scaler_a = scale_features(X_train, rng.normal(size=(20, 3)), rng.normal(size=(20, 3)))
    *_, scaler_b = scale_features(X_train, np.full((20, 3), 1e9), np.full((20, 3), -1e9))

    np.testing.assert_array_equal(scaler_a.mean_, scaler_b.mean_)
    np.testing.assert_array_equal(scaler_a.var_, scaler_b.var_)


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


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_pipeline_keeps_x_test_out_of_tuning_and_training(
    fast_settings: Settings, small_raw_data_dir: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {"tune": [], "train": [], "curves": [], "final": []}
    splits: dict[str, Any] = {}

    def spy_preprocess(*args: Any, **kwargs: Any) -> Any:
        out = preprocess(*args, **kwargs)
        splits.update(X_train=out[0], X_val=out[1], X_test=out[2])
        return out

    def spy(kind: str, fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            seen[kind].append(args)
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(pipeline, "preprocess", spy_preprocess)
    for name in ("tune_logreg", "tune_svm", "tune_rf", "tune_mlp"):
        monkeypatch.setattr(pipeline, name, spy("tune", getattr(pipeline, name)))
    for name in ("run_logreg", "run_svm", "run_rf", "run_mlp"):
        monkeypatch.setattr(pipeline, name, spy("train", getattr(pipeline, name)))
    monkeypatch.setattr(
        pipeline, "plot_learning_curve_sklearn", spy("curves", lambda *a, **k: None)
    )
    monkeypatch.setattr(pipeline, "plot_learning_curve_mlp", spy("curves", lambda *a, **k: None))
    monkeypatch.setattr(pipeline, "evaluate_on_test", spy("final", pipeline.evaluate_on_test))

    pipeline.run(fast_settings)

    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    assert len(seen["tune"]) == len(seen["train"]) == 4
    assert all(args[0] is X_train for args in seen["tune"])
    assert all(args[0] is X_train and args[2] is X_val for args in seen["train"])
    assert all(args[0] is X_train or args[1] is X_train for args in seen["curves"])
    for args in seen["tune"] + seen["train"] + seen["curves"]:
        assert not any(a is X_test for a in args if isinstance(a, np.ndarray))
    assert len(seen["final"]) == 1  # the test set is evaluated exactly once
    assert seen["final"][0][1] is X_test
