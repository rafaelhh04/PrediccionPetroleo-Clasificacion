"""scikit-learn transformers for the preprocessing steps that are fitted on data.

- :class:`FeatureEngineer` is stateless: it derives the features from the merged
  daily frame using only past observations (lags, rolling windows). It runs on the
  whole chronological frame *before* splitting, because the windows need history;
  the leakage tests prove that no feature looks at the future.
- :class:`VIFSelector` and :class:`Winsorizer` learn from the data, so they live
  inside the model :class:`~sklearn.pipeline.Pipeline` and are only ever fitted on
  training rows (also within every cross-validation fold).
"""

import warnings
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from statsmodels.stats.outliers_influence import variance_inflation_factor

from brent_forecast._types import FloatArray


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless transformer: merged daily frame -> engineered feature frame.

    ``fit`` learns nothing (it only records the output columns); ``transform``
    returns the feature columns selected by
    :func:`brent_forecast.features.preprocessing.engineer_features`.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Record the names of the engineered feature columns."""
        from brent_forecast.features.preprocessing import engineer_features

        _, self.feature_names_out_ = engineer_features(X.copy())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the engineered features (one row per input row)."""
        from brent_forecast.features.preprocessing import engineer_features

        check_is_fitted(self, "feature_names_out_")
        df, _ = engineer_features(X.copy())
        return df[self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> npt.NDArray[np.object_]:
        """Return the names of the engineered features."""
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_, dtype=object)


class VIFSelector(TransformerMixin, BaseEstimator):
    """Drop features iteratively by highest Variance Inflation Factor.

    While the largest VIF exceeds ``threshold`` the corresponding feature is
    removed and the VIFs are recomputed. A VIF above 10 means the feature is
    almost a linear combination of the others.

    Parameters
    ----------
    threshold
        Maximum VIF a surviving feature may have.
    """

    def __init__(self, threshold: float = 10.0) -> None:
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> Self:
        """Select the features using only the rows in ``X``."""
        X_arr: FloatArray = validate_data(self, X, dtype=np.float64)
        remaining = list(range(X_arr.shape[1]))
        self.dropped_vifs_: dict[int, float] = {}

        while len(remaining) > 1:
            vifs = _vifs(X_arr[:, remaining])
            worst = int(np.argmax(vifs))
            if vifs[worst] <= self.threshold:
                break
            self.dropped_vifs_[remaining[worst]] = vifs[worst]
            remaining.pop(worst)

        self.support_ = np.zeros(X_arr.shape[1], dtype=bool)
        self.support_[remaining] = True
        self.vifs_ = _vifs(X_arr[:, remaining]) if len(remaining) > 1 else np.ones(1)
        return self

    def transform(self, X: Any) -> Any:
        """Keep the selected columns."""
        check_is_fitted(self, "support_")
        X_arr: FloatArray = validate_data(self, X, dtype=np.float64, reset=False)
        return X_arr[:, self.support_]

    def get_feature_names_out(self, input_features: Any = None) -> npt.NDArray[np.object_]:
        """Return the names of the selected features."""
        check_is_fitted(self, "support_")
        names = _input_names(self, input_features)
        return np.asarray(names, dtype=object)[self.support_]


class Winsorizer(TransformerMixin, BaseEstimator):
    """Clip columns to percentiles learned on the training data.

    Parameters
    ----------
    lower, upper
        Percentile bounds as fractions (e.g. 0.01 and 0.99).
    name_contains
        Only clip columns whose name contains this substring (``"ret"`` selects
        the return features). ``None`` clips every column.
    """

    def __init__(
        self, lower: float = 0.01, upper: float = 0.99, name_contains: str | None = None
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.name_contains = name_contains

    def fit(self, X: Any, y: Any = None) -> Self:
        """Learn the clipping bounds of the selected columns."""
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError("Winsorizer requires 0 <= lower < upper <= 1.")
        X_arr: FloatArray = validate_data(self, X, dtype=np.float64)
        names = _input_names(self, None)
        if self.name_contains is None:
            columns = list(range(X_arr.shape[1]))
        else:
            columns = [i for i, name in enumerate(names) if self.name_contains in name]
        self.columns_ = np.asarray(columns, dtype=int)
        self.lower_bounds_ = np.percentile(X_arr[:, self.columns_], self.lower * 100, axis=0)
        self.upper_bounds_ = np.percentile(X_arr[:, self.columns_], self.upper * 100, axis=0)
        return self

    def transform(self, X: Any) -> Any:
        """Clip the selected columns to the learned bounds (returns a copy)."""
        check_is_fitted(self, "columns_")
        X_arr: FloatArray = validate_data(self, X, dtype=np.float64, reset=False, copy=True)
        if self.columns_.size:
            X_arr[:, self.columns_] = np.clip(
                X_arr[:, self.columns_], self.lower_bounds_, self.upper_bounds_
            )
        return X_arr

    def get_feature_names_out(self, input_features: Any = None) -> npt.NDArray[np.object_]:
        """Return the input names unchanged (clipping keeps every column)."""
        check_is_fitted(self, "columns_")
        return np.asarray(_input_names(self, input_features), dtype=object)


def _vifs(X: FloatArray) -> list[float]:
    """VIF of every column of ``X``; numerically singular columns get ``inf``."""
    vifs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels warns on near-singular designs
        for i in range(X.shape[1]):
            try:
                vifs.append(float(variance_inflation_factor(X, i)))
            except Exception:
                vifs.append(float("inf"))
    return [v if np.isfinite(v) else float("inf") for v in vifs]


def _input_names(estimator: BaseEstimator, input_features: Sequence[str] | None) -> list[str]:
    """Input feature names, falling back to ``x0, x1, ...`` like scikit-learn."""
    if input_features is not None:
        return [str(f) for f in input_features]
    names = getattr(estimator, "feature_names_in_", None)
    if names is not None:
        return [str(f) for f in names]
    return [f"x{i}" for i in range(estimator.n_features_in_)]
