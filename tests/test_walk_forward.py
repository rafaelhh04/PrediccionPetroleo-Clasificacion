"""Tests for brent_forecast.validation.walk_forward (purged walk-forward CV)."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

from brent_forecast.validation.walk_forward import PurgedWalkForwardSplit, walk_forward_predict


def _splits(cv: PurgedWalkForwardSplit, n: int) -> list[tuple[list[int], list[int]]]:
    return [(tr.tolist(), te.tolist()) for tr, te in cv.split(np.zeros((n, 1)))]


# ── geometry ──────────────────────────────────────────────────


def test_folds_mode_matches_time_series_split_without_gap() -> None:
    X = np.zeros((50, 1))

    ours = list(PurgedWalkForwardSplit(4, purge=0, embargo=0).split(X))
    sklearn = list(TimeSeriesSplit(4).split(X))

    for (tr_a, te_a), (tr_b, te_b) in zip(ours, sklearn, strict=True):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_purge_and_embargo_open_a_gap_before_each_test_window() -> None:
    splits = _splits(PurgedWalkForwardSplit(2, test_size=5, purge=1, embargo=2), 20)

    assert splits == [
        (list(range(0, 7)), list(range(10, 15))),  # rows 7-9: purge 1 + embargo 2
        (list(range(0, 12)), list(range(15, 20))),
    ]


def test_windows_mode_covers_the_tail_with_a_short_last_window() -> None:
    cv = PurgedWalkForwardSplit(None, test_size=4, first_test_index=10, purge=1)

    splits = _splits(cv, 20)

    assert [te for _, te in splits] == [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19]]
    assert [tr[-1] for tr, _ in splits] == [8, 12, 16]  # last row before each window is purged
    assert cv.get_n_splits(np.zeros((20, 1))) == 3


def test_rolling_window_keeps_only_the_most_recent_rows() -> None:
    cv = PurgedWalkForwardSplit(None, test_size=5, first_test_index=10, purge=1, max_train_size=4)

    assert [tr for tr, _ in _splits(cv, 20)] == [[5, 6, 7, 8], [10, 11, 12, 13]]


def test_gap_property_and_repr() -> None:
    cv = PurgedWalkForwardSplit(3, purge=1, embargo=5)

    assert cv.gap == 6
    assert "purge=1" in repr(cv)
    assert "embargo=5" in repr(cv)


@given(
    n=st.integers(30, 300),
    n_splits=st.integers(1, 6),
    purge=st.integers(0, 3),
    embargo=st.integers(0, 5),
    rolling=st.one_of(st.none(), st.integers(5, 50)),
)
@settings(max_examples=100, deadline=None)
def test_every_split_respects_time_order_and_the_gap(
    n: int, n_splits: int, purge: int, embargo: int, rolling: int | None
) -> None:
    cv = PurgedWalkForwardSplit(n_splits, purge=purge, embargo=embargo, max_train_size=rolling)
    try:
        splits = list(cv.split(np.zeros((n, 1))))
    except ValueError:
        return  # configuration too large for n: rejected explicitly, never silently

    assert len(splits) == n_splits
    for train, test in splits:
        assert len(train) > 0
        assert len(test) > 0
        assert train.max() + purge + embargo < test.min()  # strict gap
        assert np.all(np.diff(train) == 1)
        assert np.all(np.diff(test) == 1)
        if rolling is not None:
            assert len(train) <= rolling
    tests = np.concatenate([te for _, te in splits])
    assert len(np.unique(tests)) == len(tests)  # test windows never overlap


# ── validation of arguments ───────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_splits": None}, "n_splits must be >= 1"),
        ({"n_splits": None, "first_test_index": 5}, "test_size is required"),
        ({"purge": -1}, "purge and embargo"),
        ({"test_size": 0}, "test_size must be"),
        ({"max_train_size": 0}, "max_train_size"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PurgedWalkForwardSplit(**{"n_splits": 3, **kwargs})


def test_too_little_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot make"):
        list(PurgedWalkForwardSplit(10).split(np.zeros((5, 1))))
    with pytest.raises(ValueError, match="training rows"):
        list(PurgedWalkForwardSplit(2, test_size=5, purge=1, embargo=5).split(np.zeros((12, 1))))
    with pytest.raises(ValueError, match="outside"):
        list(
            PurgedWalkForwardSplit(None, test_size=2, first_test_index=50).split(np.zeros((10, 1)))
        )


def test_windows_mode_needs_x_to_count_splits() -> None:
    with pytest.raises(ValueError, match="X is required"):
        PurgedWalkForwardSplit(None, test_size=2, first_test_index=3).get_n_splits()


# ── scikit-learn integration ──────────────────────────────────


def test_works_as_cv_in_scikit_learn(toy_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = toy_xy
    cv = PurgedWalkForwardSplit(3, purge=1, embargo=2)

    scores = cross_val_score(LogisticRegression(), X, y, cv=cv, scoring="roc_auc")
    search = GridSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}, cv=cv, scoring="roc_auc")
    search.fit(X, y)

    assert len(scores) == 3
    assert search.n_splits_ == 3


# ── walk_forward_predict ──────────────────────────────────────


def _frame(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, 3)), columns=["a", "b", "c"], index=np.arange(100, 100 + n)
    )
    y = pd.Series((X["a"] + rng.normal(size=n) > 0).astype(float), index=X.index)
    return X, y


def test_walk_forward_predict_covers_every_test_row_once() -> None:
    X, y = _frame()
    cv = PurgedWalkForwardSplit(None, test_size=40, first_test_index=180, purge=1, embargo=3)

    preds = walk_forward_predict(LogisticRegression(), X, y, cv)

    assert preds.proba.index.tolist() == X.index[180:].tolist()
    assert preds.window.tolist() == [0] * 40 + [1] * 40 + [2] * 40
    assert ((preds.proba > 0) & (preds.proba < 1)).all()


def test_predictions_never_depend_on_their_own_window_or_later_rows() -> None:
    """The walk-forward analogue of the future-permutation test."""
    X, y = _frame()
    cv = PurgedWalkForwardSplit(None, test_size=40, first_test_index=180, purge=1, embargo=3)
    base = walk_forward_predict(LogisticRegression(), X, y, cv)

    for window_start in (180, 220, 260):
        X_bad, y_bad = X.copy(), y.copy()
        X_bad.iloc[window_start:] = 1e6  # features of this window and every later row
        y_bad.iloc[window_start - 1 :] = 1.0 - y_bad.iloc[window_start - 1 :]  # purged labels too

        changed = walk_forward_predict(LogisticRegression(), X_bad, y_bad, cv)

        before = base.proba.index < X.index[window_start]
        pd.testing.assert_series_equal(changed.proba[before], base.proba[before])


def test_refit_sees_more_data_in_later_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _frame()
    cv = PurgedWalkForwardSplit(None, test_size=40, first_test_index=180, purge=1, embargo=3)
    sizes: list[int] = []
    original_fit = LogisticRegression.fit

    def spy_fit(self, X_fit, y_fit):  # type: ignore[no-untyped-def]
        sizes.append(len(X_fit))
        return original_fit(self, X_fit, y_fit)

    monkeypatch.setattr(LogisticRegression, "fit", spy_fit)

    walk_forward_predict(LogisticRegression(), X, y, cv)

    assert sizes == [176, 216, 256]  # window start - purge - embargo


def test_without_a_gap_the_boundary_label_leaks_into_the_next_window() -> None:
    """Control for the test above: with purge = embargo = 0 the check would fail."""
    X, y = _frame()
    cv = PurgedWalkForwardSplit(None, test_size=40, first_test_index=180, purge=0, embargo=0)
    base = walk_forward_predict(LogisticRegression(C=100.0), X, y, cv)
    y_bad = y.copy()
    y_bad.iloc[179:] = 1.0 - y_bad.iloc[179:]  # row 179's label is realised on day 180

    changed = walk_forward_predict(LogisticRegression(C=100.0), X, y_bad, cv)

    first_window = base.window == 0
    assert not np.allclose(changed.proba[first_window], base.proba[first_window])
