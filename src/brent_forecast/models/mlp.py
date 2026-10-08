"""scikit-learn estimator around the pure-NumPy MLP of :mod:`.neural_network`.

The network itself (forward pass, backpropagation, dropout, SGD) stays in
pure NumPy; this class only adapts it to the estimator API so that it can be
tuned, cross-validated, put in a :class:`~sklearn.pipeline.Pipeline` and
serialised like any other model.
"""

from typing import Any, Literal, Self

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from brent_forecast._types import FloatArray
from brent_forecast.models.neural_network import EpochRecord, Params, predict_proba, train_mlp


class NumpyMLPClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier: two ReLU hidden layers with inverted dropout and a sigmoid output.

    Trained with mini-batch SGD. Early stopping monitors a validation metric on
    the *last* ``validation_fraction`` of the training samples (chronological
    hold-out, no shuffling, so it never looks at later data than it trains on
    in a time series).

    Parameters
    ----------
    hidden_1, hidden_2
        Sizes of the hidden layers.
    dropout_p
        Dropout probability of both hidden layers.
    learning_rate
        SGD step size.
    batch_size
        Mini-batch size.
    max_epochs
        Maximum number of epochs.
    patience
        Epochs without improvement of the validation AUC before stopping.
    validation_fraction
        Fraction of the (last) training samples held out for early stopping;
        0 disables early stopping and keeps the weights of the last epoch.
    early_stopping_metric
        ``"loss"`` (validation BCE, default) or ``"auc"`` (validation ROC AUC).
        The loss is preferred: the AUC of a small hold-out saturates or is very
        noisy, which stops training too early.
    random_state
        Seed for initialisation, dropout masks and shuffling.
    """

    def __init__(
        self,
        hidden_1: int = 64,
        hidden_2: int = 32,
        dropout_p: float = 0.2,
        learning_rate: float = 0.01,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        validation_fraction: float = 0.1,
        early_stopping_metric: Literal["loss", "auc"] = "loss",
        random_state: int | None = None,
    ) -> None:
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.early_stopping_metric = early_stopping_metric
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> Self:
        """Train the network on ``(X, y)``; ``y`` must contain exactly two classes."""
        X_arr, y_arr = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y_arr)
        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(
                f"Only binary classification is supported. {type(self).__name__} got "
                f"{len(self.classes_)} classes."
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        y01 = (y_arr == self.classes_[1]).astype(np.float64)

        n_val = int(round(len(y01) * self.validation_fraction))
        X_tr, y_tr, X_val, y_val = X_arr, y01, X_arr, y01
        use_early_stopping = n_val > 0 and len(np.unique(y01[-n_val:])) == 2
        if use_early_stopping:
            X_tr, y_tr = X_arr[:-n_val], y01[:-n_val]
            X_val, y_val = X_arr[-n_val:], y01[-n_val:]

        seed = 0 if self.random_state is None else int(self.random_state)
        weights, history = train_mlp(
            X_tr,
            y_tr,
            X_val,
            y_val,
            hidden_1=self.hidden_1,
            hidden_2=self.hidden_2,
            dropout_p=self.dropout_p,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience if use_early_stopping else self.max_epochs,
            random_state=seed,
            verbose=False,
            # Without a usable hold-out, keep the weights of the last epoch.
            restore_best=use_early_stopping,
            monitor=self.early_stopping_metric,
        )
        self.weights_: Params = weights
        self.history_: list[EpochRecord] = history
        self.n_epochs_ = len(history)
        return self

    def predict_proba(self, X: Any) -> FloatArray:
        """Return ``[P(class 0), P(class 1)]`` for every row."""
        check_is_fitted(self, "weights_")
        X_arr: FloatArray = validate_data(self, X, dtype=np.float64, reset=False)
        p1 = predict_proba(X_arr, self.weights_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> Any:
        """Predict the class with probability >= 0.5."""
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def __sklearn_tags__(self) -> Any:
        """Declare the estimator as binary-only."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
