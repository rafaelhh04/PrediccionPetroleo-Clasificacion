"""Tests for brent_forecast.evaluation.backtest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brent_forecast.evaluation.backtest import (
    max_drawdown,
    performance,
    plot_equity,
    positions,
    sharpe,
    sharpe_difference,
    strategy_returns,
)


def test_positions_long_flat_and_long_short() -> None:
    proba = pd.Series([0.2, 0.5, 0.8])

    assert positions(proba, 0.5, allow_short=False).tolist() == [0.0, 1.0, 1.0]
    assert positions(proba, 0.6, allow_short=True).tolist() == [-1.0, -1.0, 1.0]


def test_strategy_returns_charge_costs_on_every_change_of_position() -> None:
    position = pd.Series([1.0, 1.0, 0.0, -1.0])
    next_return = pd.Series([0.01, -0.02, 0.03, -0.01])

    out = strategy_returns(position, next_return, cost_bps=10)

    np.testing.assert_allclose(out["gross"], [0.01, -0.02, 0.0, 0.01])
    np.testing.assert_allclose(out["cost"], [0.001, 0.0, 0.001, 0.001])  # entry, exit, short
    np.testing.assert_allclose(out["net"], out["gross"] - out["cost"])


def test_a_perfect_forecast_earns_every_up_day() -> None:
    rng = np.random.default_rng(0)
    next_return = pd.Series(rng.normal(0, 0.02, 500))
    oracle = (next_return > 0).astype(float)

    perfect = performance(strategy_returns(oracle, next_return, 0.0), 252)

    assert perfect["hit_ratio"] == 1.0
    assert perfect["max_drawdown"] == 0.0
    assert perfect["sharpe"] > 10
    assert perfect["exposure"] == pytest.approx(oracle.mean())


def test_costs_can_destroy_a_high_turnover_strategy() -> None:
    rng = np.random.default_rng(1)
    next_return = pd.Series(rng.normal(0.0005, 0.02, 1000))
    flip = pd.Series(np.arange(1000) % 2, dtype=float)  # trades every day

    free = performance(strategy_returns(flip, next_return, 0.0), 252)
    costly = performance(strategy_returns(flip, next_return, 50.0), 252)

    assert costly["n_trades"] == 999
    assert costly["costs"] == pytest.approx(999 * 0.005)
    assert costly["sharpe"] < free["sharpe"] - 1


def test_buy_and_hold_statistics() -> None:
    next_return = pd.Series([0.1, -0.5, 0.2, 0.0])
    always_long = pd.Series(1.0, index=next_return.index)

    stats = performance(strategy_returns(always_long, next_return, 0.0), 4)

    assert stats["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert stats["cagr"] == pytest.approx(stats["total_return"])  # exactly one "year"
    assert stats["max_drawdown"] == pytest.approx(-0.5)
    assert stats["hit_ratio"] == 0.5
    assert stats["n_trades"] == 1


def test_a_strategy_that_never_trades() -> None:
    flat = pd.Series(0.0, index=range(10))

    stats = performance(strategy_returns(flat, pd.Series(0.01, index=range(10)), 5.0), 252)

    assert stats["hit_ratio"] is None
    assert stats["sharpe"] == 0.0
    assert stats["exposure"] == 0.0
    assert stats["n_trades"] == 0


def test_sharpe_and_drawdown_helpers() -> None:
    net = np.array([0.01, -0.005, 0.02, 0.0])

    assert sharpe(net, 252) == pytest.approx(net.mean() / net.std(ddof=1) * np.sqrt(252))
    assert sharpe_difference(net, net) == 0.0
    assert sharpe_difference(net, 2 * net, periods_per_year=4) == pytest.approx(0.0)
    assert max_drawdown(np.array([0.1, -0.1, -0.1, 0.5])) == pytest.approx(0.9 * 0.9 - 1)
    assert max_drawdown(np.array([-0.2])) == pytest.approx(-0.2)


def test_plot_equity(plots_dir: Path) -> None:
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=20))
    net = pd.Series(np.linspace(-0.01, 0.01, 20))

    path = plot_equity(dates, {"A": net, "B": -net}, plots_dir)

    assert path.name == "equity_curves.png"
    assert path.stat().st_size > 0
