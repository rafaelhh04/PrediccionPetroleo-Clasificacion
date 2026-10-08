"""Economic backtest of the out-of-sample predictions.

A classifier is only useful if its signal survives trading costs. Each
candidate is turned into a **long/flat** strategy (optionally long/short):
on day ``t`` the position for ``t -> t+1`` is long if ``P(up) >= threshold``
and flat (or short) otherwise; it earns ``position * next_return`` and pays
``cost_bps`` basis points per unit of turnover (each change of position).
Positions are decided with information available at the close of ``t``, the
same information the prediction uses.

The reference is buy & hold (always long, one entry cost). Sharpe ratios use
a zero risk-free rate and are annualised with ``periods_per_year``.
"""

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brent_forecast.evaluation.metrics import save_figure


def positions(proba: pd.Series, threshold: float, *, allow_short: bool) -> pd.Series:
    """Position for the next day: 1 (long), 0 (flat) or -1 (short, if allowed)."""
    down = -1.0 if allow_short else 0.0
    return pd.Series(np.where(proba >= threshold, 1.0, down), index=proba.index)


def strategy_returns(position: pd.Series, next_return: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Daily gross return, trading cost and net return of a position series.

    The portfolio starts flat, so entering the first position is a trade.
    """
    turnover = _turnover(position)
    gross = position * next_return
    cost = turnover * cost_bps / 10_000
    return pd.DataFrame({"position": position, "gross": gross, "cost": cost, "net": gross - cost})


def _turnover(position: pd.Series) -> pd.Series:
    """Absolute change of position each day, starting from flat."""
    return position.diff().fillna(position.iloc[0]).abs()


def sharpe(net: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (zero risk-free rate); 0 for a zero-variance series."""
    std = float(np.std(net, ddof=1))
    return 0.0 if std < 1e-12 else float(np.mean(net) / std * np.sqrt(periods_per_year))


def sharpe_difference(net_1: np.ndarray, net_2: np.ndarray, periods_per_year: int = 252) -> float:
    """``sharpe(net_1) - sharpe(net_2)``, for the paired block bootstrap."""
    return sharpe(net_1, periods_per_year) - sharpe(net_2, periods_per_year)


def max_drawdown(net: np.ndarray) -> float:
    """Largest peak-to-trough fall of the compounded equity curve (a negative fraction)."""
    equity = np.cumprod(1 + np.asarray(net, dtype=float))
    peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    return float(np.min(equity / peak - 1))


def performance(returns: pd.DataFrame, periods_per_year: int) -> dict[str, float | None]:
    """Summary statistics of :func:`strategy_returns` output.

    ``hit_ratio`` is the share of days in the market with a positive gross
    return; it is ``None`` for a strategy that never trades.
    """
    net = returns["net"].to_numpy()
    in_market = returns["position"].to_numpy() != 0
    total = float(np.prod(1 + net) - 1)
    years = len(net) / periods_per_year
    hits = returns["gross"].to_numpy()[in_market] > 0
    return {
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0,
        "volatility": float(np.std(net, ddof=1) * np.sqrt(periods_per_year)),
        "sharpe": sharpe(net, periods_per_year),
        "max_drawdown": max_drawdown(net),
        "hit_ratio": float(hits.mean()) if hits.size else None,
        "exposure": float(in_market.mean()),
        "n_trades": float((_turnover(returns["position"]) > 0).sum()),
        "costs": float(returns["cost"].sum()),
    }


def plot_equity(dates: pd.Series, net_returns: Mapping[str, pd.Series], plots_dir: Path) -> Path:
    """Compounded equity curves (growth of 1) of every strategy."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = pd.to_datetime(dates).to_numpy()
    for name, net in net_returns.items():
        ax.plot(x, np.cumprod(1 + net.to_numpy()), label=name, linewidth=1.5)
    ax.axhline(1.0, linestyle="--", color="gray")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 (net of costs)")
    ax.set_title("Out-of-sample equity curves")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_figure(fig, plots_dir, "equity_curves.png")
