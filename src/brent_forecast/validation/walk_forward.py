"""Walk-forward validation with purging and embargo.

Every split trains only on observations that are strictly older than its test
window, with a gap between both:

``purge``
    The label of day ``t`` is the direction of day ``t + 1``, so the last
    ``purge`` training rows before a test window have labels that are realised
    *inside* that window. They are removed (López de Prado, *Advances in
    Financial Machine Learning*, ch. 7). ``purge = 1`` for a one-day horizon.
``embargo``
    Extra rows dropped after the purge. Features are built from rolling
    windows (up to 30 trading days), so neighbouring days are strongly
    correlated; the embargo keeps the most recent, near-duplicate training
    rows away from the first test days.

The splitter follows scikit-learn's CV protocol (``split`` / ``get_n_splits``),
so it plugs into ``GridSearchCV`` and ``learning_curve``. It has two modes:

- **folds** (``n_splits``): like ``TimeSeriesSplit``, the last ``n_splits``
  blocks of ``test_size`` rows are the test folds (hyperparameter tuning).
- **windows** (``first_test_index``): from that row to the end, consecutive
  windows of ``test_size`` rows (the last one may be shorter); the model is
  refitted before each window (out-of-sample evaluation).

Training sets are expanding by default; ``max_train_size`` turns them into a
rolling window of the most recent rows.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import clone

IndexArray = npt.NDArray[np.intp]


@dataclass(frozen=True)
class Window:
    """Row positions of one walk-forward step."""

    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive

    @property
    def train(self) -> IndexArray:
        """Training row positions."""
        return np.arange(self.train_start, self.train_end)

    @property
    def test(self) -> IndexArray:
        """Test row positions."""
        return np.arange(self.test_start, self.test_end)


class PurgedWalkForwardSplit:
    """Walk-forward cross-validator with purge and embargo.

    Parameters
    ----------
    n_splits
        Number of test folds at the end of the data (folds mode). Ignored when
        ``first_test_index`` is given.
    test_size
        Rows per test fold / window. In folds mode it defaults to
        ``n_samples // (n_splits + 1)`` like ``TimeSeriesSplit``.
    first_test_index
        Row where the first test window starts (windows mode).
    purge
        Training rows removed right before every test window (label horizon).
    embargo
        Additional rows removed after the purge.
    max_train_size
        If set, train on at most this many most recent rows (rolling window).
    min_train_size
        Minimum training rows a split must have; otherwise ``split`` raises.
    """

    def __init__(
        self,
        n_splits: int | None = 5,
        *,
        test_size: int | None = None,
        first_test_index: int | None = None,
        purge: int = 1,
        embargo: int = 0,
        max_train_size: int | None = None,
        min_train_size: int = 1,
    ) -> None:
        if first_test_index is None and (n_splits is None or n_splits < 1):
            raise ValueError("n_splits must be >= 1 when first_test_index is not given.")
        if first_test_index is not None and test_size is None:
            raise ValueError("test_size is required when first_test_index is given.")
        if purge < 0 or embargo < 0:
            raise ValueError("purge and embargo must be >= 0.")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size must be >= 1.")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError("max_train_size must be >= 1.")
        self.n_splits = n_splits
        self.test_size = test_size
        self.first_test_index = first_test_index
        self.purge = purge
        self.embargo = embargo
        self.max_train_size = max_train_size
        self.min_train_size = min_train_size

    @property
    def gap(self) -> int:
        """Rows between the end of training and the start of each test window."""
        return self.purge + self.embargo

    def windows(self, n_samples: int) -> list[Window]:
        """Compute the walk-forward windows for ``n_samples`` chronologically ordered rows."""
        if self.first_test_index is not None:
            first, size = self.first_test_index, self._window_size()
            if not 0 < first < n_samples:
                raise ValueError(f"first_test_index={first} outside (0, {n_samples}).")
            starts = list(range(first, n_samples, size))
        else:
            n_splits = self._n_folds()
            size = self.test_size or n_samples // (n_splits + 1)
            first = n_samples - n_splits * size
            if size < 1 or first < 1:
                raise ValueError(
                    f"Cannot make {n_splits} test folds of {size} rows from {n_samples} samples."
                )
            starts = [first + i * size for i in range(n_splits)]

        windows = []
        for start in starts:
            train_end = start - self.gap
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            if train_end - train_start < self.min_train_size:
                raise ValueError(
                    f"Test window starting at row {start} leaves {max(train_end - train_start, 0)} "
                    f"training rows (< min_train_size={self.min_train_size})."
                )
            windows.append(Window(train_start, train_end, start, min(start + size, n_samples)))
        return windows

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[IndexArray, IndexArray]]:
        """Yield ``(train_positions, test_positions)`` for every window."""
        for window in self.windows(len(X)):
            yield window.train, window.test

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of splits (windows mode needs ``X`` to count the rows)."""
        if self.first_test_index is None:
            return self._n_folds()
        if X is None:
            raise ValueError("X is required to count the windows when first_test_index is set.")
        return math.ceil((len(X) - self.first_test_index) / self._window_size())

    def _n_folds(self) -> int:
        if self.n_splits is None:  # excluded by __init__ in folds mode
            raise ValueError("n_splits is required in folds mode.")
        return self.n_splits

    def _window_size(self) -> int:
        if self.test_size is None:  # excluded by __init__ in windows mode
            raise ValueError("test_size is required in windows mode.")
        return self.test_size

    def __repr__(self) -> str:
        """Readable summary of the configuration."""
        return (
            f"{type(self).__name__}(n_splits={self.n_splits}, test_size={self.test_size}, "
            f"first_test_index={self.first_test_index}, purge={self.purge}, "
            f"embargo={self.embargo}, max_train_size={self.max_train_size})"
        )


@dataclass(frozen=True)
class WalkForwardPredictions:
    """Out-of-sample predictions of one model over every test window."""

    proba: pd.Series
    """``P(y = 1)``, indexed like the evaluated rows."""
    window: pd.Series
    """Window number (0, 1, ...) of every predicted row."""


def walk_forward_predict(
    estimator: Any, X: pd.DataFrame, y: pd.Series, cv: PurgedWalkForwardSplit
) -> WalkForwardPredictions:
    """Refit a clone of ``estimator`` before every window and predict that window.

    Only rows older than ``window.test_start - cv.gap`` are used to fit the
    model that predicts the window, so every prediction is out of sample.
    """
    probas, windows = [], []
    for k, (train_idx, test_idx) in enumerate(cv.split(X)):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        probas.append(pd.Series(proba, index=X.index[test_idx]))
        windows.append(pd.Series(k, index=X.index[test_idx]))
    return WalkForwardPredictions(proba=pd.concat(probas), window=pd.concat(windows))
