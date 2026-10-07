"""Numerical gradient checking and property-based tests for the NumPy MLP.

The analytical gradients of ``backward`` are compared with central finite
differences of ``binary_cross_entropy(forward(...))`` for every parameter,
with dropout disabled and with a fixed dropout mask. Relative error per
parameter tensor: ``||g_num - g_ana|| / (||g_num|| + ||g_ana||) < 1e-6``.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from brent_forecast.models.neural_network import (
    Params,
    backward,
    binary_cross_entropy,
    forward,
    initialize_parameters,
    predict_proba,
    sgd_step,
    sigmoid,
    train_mlp,
)

PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
EPS = 1e-6
TOLERANCE = 1e-6


def _problem(seed: int, n: int = 16, n_in: int = 4, h1: int = 6, h2: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_in))
    y = rng.integers(0, 2, n).astype(np.float64)
    params = initialize_parameters(n_in, h1, h2, np.random.RandomState(seed))
    return X, y, params


def _kink_free_problem(seed: int) -> tuple[np.ndarray, np.ndarray, Params]:
    """Problem whose ReLU pre-activations all stay far from 0 (the non-differentiable point).

    With the default zero biases, a sample whose first-layer units are all
    inactive yields ``Z2 = 0`` exactly; there the analytic derivative of ReLU is 0
    while a central difference returns 0.5, a property of finite differences
    rather than a backprop bug. Small random biases avoid those points.
    """
    X, y, params = _problem(seed)
    rng = np.random.default_rng(seed + 1000)
    for name in ("b1", "b2", "b3"):
        params[name] = rng.normal(0, 0.1, params[name].shape)
    _, cache = forward(X, params, False, np.random.RandomState(0), 0.0)
    assert np.abs(cache["Z1"]).min() > 100 * EPS
    assert np.abs(cache["Z2"]).min() > 100 * EPS
    return X, y, params


def _loss(X: np.ndarray, y: np.ndarray, params: Params, dropout_p: float, mask_seed: int) -> float:
    # A fresh RandomState with the same seed draws the same dropout masks: the
    # masks only depend on the activation shapes, not on the parameter values.
    training = dropout_p > 0
    A3, _ = forward(X, params, training, np.random.RandomState(mask_seed), dropout_p)
    return binary_cross_entropy(y, A3)


def _numerical_gradients(
    X: np.ndarray, y: np.ndarray, params: Params, dropout_p: float, mask_seed: int
) -> Params:
    grads: Params = {}
    for name in PARAM_NAMES:
        grad = np.zeros_like(params[name])
        it = np.nditer(params[name], flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = params[name][idx]
            params[name][idx] = original + EPS
            plus = _loss(X, y, params, dropout_p, mask_seed)
            params[name][idx] = original - EPS
            minus = _loss(X, y, params, dropout_p, mask_seed)
            params[name][idx] = original
            grad[idx] = (plus - minus) / (2 * EPS)
        grads[f"d{name}"] = grad
    return grads


def _relative_errors(
    X: np.ndarray, y: np.ndarray, params: Params, dropout_p: float, mask_seed: int = 0
) -> dict[str, float]:
    training = dropout_p > 0
    _, cache = forward(X, params, training, np.random.RandomState(mask_seed), dropout_p)
    analytic = backward(y, cache, params, dropout_p)
    numeric = _numerical_gradients(X, y, params, dropout_p, mask_seed)
    errors = {}
    for name in PARAM_NAMES:
        a, n = analytic[f"d{name}"], numeric[f"d{name}"]
        assert a.shape == n.shape == params[name].shape
        errors[name] = float(np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n)))
    return errors


# ── gradient checking ─────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradients_match_finite_differences_without_dropout(seed: int) -> None:
    X, y, params = _kink_free_problem(seed)

    errors = _relative_errors(X, y, params, dropout_p=0.0)

    assert max(errors.values()) < TOLERANCE, errors


@pytest.mark.parametrize(("seed", "dropout_p"), [(0, 0.2), (1, 0.5)])
def test_gradients_match_finite_differences_with_a_fixed_dropout_mask(
    seed: int, dropout_p: float
) -> None:
    X, y, params = _kink_free_problem(seed)

    errors = _relative_errors(X, y, params, dropout_p=dropout_p, mask_seed=seed + 10)

    assert max(errors.values()) < TOLERANCE, errors


def test_gradient_check_detects_a_wrong_gradient() -> None:
    """Sanity check of the checker itself: a perturbed gradient must fail."""
    X, y, params = _kink_free_problem(0)
    _, cache = forward(X, params, False, np.random.RandomState(0), 0.0)
    analytic = backward(y, cache, params, 0.0)
    numeric = _numerical_gradients(X, y, params, 0.0, 0)

    wrong = analytic["dW2"] * 1.01
    error = np.linalg.norm(wrong - numeric["dW2"]) / (
        np.linalg.norm(wrong) + np.linalg.norm(numeric["dW2"])
    )

    assert error > TOLERANCE


# ── properties ────────────────────────────────────────────────

finite_floats = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


@given(arrays(np.float64, st.integers(1, 50), elements=finite_floats))
def test_sigmoid_is_bounded_monotone_and_stable(z: np.ndarray) -> None:
    with np.errstate(over="raise", invalid="raise"):
        s = sigmoid(z)

    assert np.isfinite(s).all()
    assert ((s >= 0) & (s <= 1)).all()
    moderate = np.abs(z) < 30
    assert ((s[moderate] > 0) & (s[moderate] < 1)).all()
    np.testing.assert_allclose(sigmoid(-z), 1 - s, atol=1e-12)
    order = np.argsort(z)
    assert (np.diff(s[order]) >= 0).all()


@given(
    arrays(np.float64, st.integers(1, 50), elements=st.floats(0, 1)),
    st.data(),
)
def test_bce_is_non_negative_and_finite(y_pred: np.ndarray, data: st.DataObject) -> None:
    y_true = data.draw(arrays(np.float64, y_pred.shape, elements=st.sampled_from([0.0, 1.0])))

    loss = binary_cross_entropy(y_true, y_pred)

    assert np.isfinite(loss)
    assert loss >= 0


@given(
    n=st.integers(1, 20),
    n_in=st.integers(1, 8),
    h1=st.integers(1, 8),
    h2=st.integers(1, 8),
    dropout_p=st.sampled_from([0.0, 0.3]),
    seed=st.integers(0, 2**31 - 1),
)
@settings(max_examples=50, deadline=None)
def test_forward_backward_shapes(
    n: int, n_in: int, h1: int, h2: int, dropout_p: float, seed: int
) -> None:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, n_in))
    y = rng.randint(0, 2, n).astype(np.float64)
    params = initialize_parameters(n_in, h1, h2, rng)

    A3, cache = forward(X, params, dropout_p > 0, rng, dropout_p)
    grads = backward(y, cache, params, dropout_p)

    assert A3.shape == (n, 1)
    assert ((A3 > 0) & (A3 < 1)).all()
    for name in PARAM_NAMES:
        assert grads[f"d{name}"].shape == params[name].shape
        assert np.isfinite(grads[f"d{name}"]).all()


@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_predict_proba_is_deterministic(seed: int) -> None:
    X, _, params = _problem(seed % 1000)

    first = predict_proba(X, params)
    np.random.seed(seed)  # global RNG state must not matter
    second = predict_proba(X, params)

    np.testing.assert_array_equal(first, second)


@given(seed=st.integers(0, 2**31 - 1))
@settings(max_examples=25, deadline=None)
def test_sgd_step_with_zero_learning_rate_is_a_no_op(seed: int) -> None:
    X, y, params = _problem(seed % 1000)
    before = {k: v.copy() for k, v in params.items()}
    _, cache = forward(X, params, False, np.random.RandomState(0), 0.0)

    after = sgd_step(params, backward(y, cache, params, 0.0), lr=0.0)

    for name in PARAM_NAMES:
        np.testing.assert_array_equal(after[name], before[name])


def _toy_data(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] + rng.normal(size=120) > 0).astype(np.float64)
    return X[:80], y[:80], X[80:], y[80:]


TRAIN_KWARGS = {"hidden_1": 4, "hidden_2": 3, "dropout_p": 0.2, "batch_size": 16}


@given(random_state=st.integers(0, 10_000))
@settings(max_examples=10, deadline=None)
def test_same_random_state_gives_same_weights(random_state: int) -> None:
    X_tr, y_tr, X_va, y_va = _toy_data(0)
    kwargs = {**TRAIN_KWARGS, "learning_rate": 0.05, "max_epochs": 5, "patience": 5}

    w1, h1 = train_mlp(X_tr, y_tr, X_va, y_va, random_state=random_state, verbose=False, **kwargs)
    w2, h2 = train_mlp(X_tr, y_tr, X_va, y_va, random_state=random_state, verbose=False, **kwargs)

    assert h1 == h2
    for name in PARAM_NAMES:
        np.testing.assert_array_equal(w1[name], w2[name])


@given(
    patience=st.integers(1, 6),
    learning_rate=st.sampled_from([0.0, 0.01, 0.5]),
    seed=st.integers(0, 50),
)
@settings(max_examples=30, deadline=None)
def test_early_stopping_respects_patience(patience: int, learning_rate: float, seed: int) -> None:
    X_tr, y_tr, X_va, y_va = _toy_data(seed)
    max_epochs = 25

    _, history = train_mlp(
        X_tr,
        y_tr,
        X_va,
        y_va,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        random_state=seed,
        verbose=False,
        **TRAIN_KWARGS,
    )

    aucs = [h["val_auc"] for h in history]
    best_epoch = int(np.argmax(aucs)) + 1  # first epoch reaching the best AUC
    assert [h["epoch"] for h in history] == list(range(1, len(history) + 1))
    assert len(history) <= max_epochs
    if len(history) < max_epochs:  # stopped early: exactly `patience` epochs without improvement
        assert len(history) - best_epoch == patience
    else:
        assert len(history) - best_epoch <= patience
