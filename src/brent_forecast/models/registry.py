"""Model registry: configuration key -> scikit-learn pipeline and search space.

Every model is the same pipeline with a different final estimator::

    VIFSelector -> Winsorizer (return features) -> StandardScaler -> model

so that each preprocessing step is fitted on the training rows of whatever
split it is given (train, a CV fold or a walk-forward window), never on more.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from brent_forecast.config import PreprocessingSettings
from brent_forecast.features.transformers import VIFSelector, Winsorizer
from brent_forecast.models.mlp import NumpyMLPClassifier

MODEL_NAMES: Final[dict[str, str]] = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (RBF)",
    "random_forest": "Random Forest",
    "mlp": "MLP NumPy",
    "lightgbm": "LightGBM",
}
"""Configuration key -> display name, in reporting order."""

# Name fragment that identifies the return features to winsorise.
RETURN_FEATURES = "ret"


def build_estimator(key: str, params: Mapping[str, Any], *, for_search: bool = False) -> Any:
    """Instantiate the final estimator of model ``key``.

    The RBF SVM has no native probabilities: the final model is wrapped in a
    sigmoid :class:`~sklearn.calibration.CalibratedClassifierCV` (Platt scaling
    on internal folds of the training data, replacing the deprecated
    ``SVC(probability=True)``). During the hyperparameter search
    (``for_search=True``) the bare SVC is used: ROC AUC only needs its
    ``decision_function``, which is ~5x cheaper.
    """
    if key == "logistic_regression":
        return LogisticRegression(**params)
    if key == "svm":
        svc = SVC(**params)
        if for_search:
            return svc
        return CalibratedClassifierCV(svc, method="sigmoid", ensemble=False)
    if key == "random_forest":
        return RandomForestClassifier(**params)
    if key == "mlp":
        return NumpyMLPClassifier(**params)
    if key == "lightgbm":
        return LGBMClassifier(**params)
    raise KeyError(f"Unknown model {key!r}; choose from {sorted(MODEL_NAMES)}")


def build_pipeline(
    key: str,
    params: Mapping[str, Any],
    preprocessing: PreprocessingSettings,
    *,
    for_search: bool = False,
) -> Pipeline:
    """Full preprocessing + model pipeline for model ``key``."""
    steps: list[tuple[str, BaseEstimator]] = [
        ("vif", VIFSelector(threshold=preprocessing.vif_threshold)),
        (
            "winsor",
            Winsorizer(
                lower=preprocessing.winsor_lower,
                upper=preprocessing.winsor_upper,
                name_contains=RETURN_FEATURES,
            ),
        ),
        ("scale", StandardScaler()),
        ("model", build_estimator(key, params, for_search=for_search)),
    ]
    return Pipeline(steps).set_output(transform="pandas")


def param_grid(key: str, grid: Mapping[str, Sequence[Any]]) -> list[dict[str, list[Any]]]:
    """Translate a configuration grid into a ``GridSearchCV`` grid on the ``model`` step.

    The MLP grid uses ``hidden_pair: [[h1, h2], ...]`` to tie both layer sizes
    together; it expands into one sub-grid per pair.
    """
    if key == "mlp" and "hidden_pair" in grid:
        rest = {f"model__{k}": list(v) for k, v in grid.items() if k != "hidden_pair"}
        return [
            {"model__hidden_1": [int(h1)], "model__hidden_2": [int(h2)], **rest}
            for h1, h2 in grid["hidden_pair"]
        ]
    return [{f"model__{k}": list(v) for k, v in grid.items()}]


def model_params(best: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the ``model__`` prefix from search results."""
    return {k.removeprefix("model__"): v for k, v in best.items()}
