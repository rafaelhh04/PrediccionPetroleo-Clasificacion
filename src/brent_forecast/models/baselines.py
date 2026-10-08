"""Naive baselines every model must beat to claim any predictive value.

All of them are scikit-learn classifiers so that they go through exactly the
same walk-forward evaluation as the real models. They receive the raw
(unscaled) feature frame:

- **Majority class**: always the training class frequency (``P(up)`` constant).
- **Persistence**: tomorrow repeats today — up if today's log return
  (``lag_ret_1``) is positive.
- **Stratified random**: random guesses with the training class frequencies.
- **Buy & hold**: always "up" (the trading counterpart is staying long).

A constant score has ROC AUC 0.5 by construction; those baselines matter for
accuracy, calibration and, above all, the economic backtest.
"""

from typing import Any, Final, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from brent_forecast._types import FloatArray

BASELINE_NAMES: Final[dict[str, str]] = {
    "majority": "Majority class",
    "persistence": "Persistence",
    "stratified": "Stratified random",
    "buy_and_hold": "Buy & hold",
}
"""Configuration key -> display name of every baseline, in reporting order."""

PERSISTENCE_FEATURE = "lag_ret_1"
"""Today's log return, ``log(price_t / price_{t-1})``."""


class PersistenceClassifier(ClassifierMixin, BaseEstimator):
    """Predict that tomorrow's direction equals today's.

    Parameters
    ----------
    column
        Feature holding today's return; it must be present in the input frame.
    """

    def __init__(self, column: str = PERSISTENCE_FEATURE) -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: Any) -> Self:
        """Record the classes; there is nothing to learn."""
        y_arr = np.asarray(y)
        check_classification_targets(y_arr)
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported.")
        if self.column not in X.columns:
            raise ValueError(f"Column {self.column!r} not found in the input frame.")
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> FloatArray:
        """Return ``[0, 1]`` for rows whose return today is positive, ``[1, 0]`` otherwise."""
        check_is_fitted(self, "classes_")
        up = (X[self.column].to_numpy(dtype=float) > 0).astype(float)
        return np.column_stack([1.0 - up, up])

    def predict(self, X: pd.DataFrame) -> Any:
        """Return the class of today's direction."""
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def build_baseline(key: str, seed: int) -> Any:
    """Instantiate baseline ``key``."""
    if key == "majority":
        return DummyClassifier(strategy="prior")
    if key == "persistence":
        return PersistenceClassifier()
    if key == "stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if key == "buy_and_hold":
        return DummyClassifier(strategy="constant", constant=1)
    raise KeyError(f"Unknown baseline {key!r}; choose from {sorted(BASELINE_NAMES)}")
