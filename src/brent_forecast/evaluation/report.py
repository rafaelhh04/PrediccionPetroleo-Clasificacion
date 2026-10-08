"""Statistical report of a training run (``brent report``).

Reads the artefacts written by ``brent train`` (``predictions.csv`` and
``metrics.json``), so the report is reproducible without retraining:

1. Per candidate: out-of-sample AUC and accuracy with block-bootstrap
   confidence intervals, binomial test of accuracy against the
   no-information rate, Brier score and Brier skill score against the
   majority-class (climatological) forecast.
2. Every model against the reference baseline (the baseline with the highest
   out-of-sample AUC): AUC difference with a paired block-bootstrap interval,
   DeLong p-value and Holm-adjusted p-value.
3. Economic backtest: every candidate as a long/flat strategy net of costs,
   against buy & hold, with a paired block-bootstrap interval of the Sharpe
   ratio difference.
4. A conclusion derived mechanically from those numbers.

Every bootstrap uses the same seed, so all candidates are resampled on the
same days (paired comparison).
"""

import json
import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from brent_forecast.config import BacktestSettings, EvaluationSettings, Settings
from brent_forecast.evaluation.backtest import (
    performance,
    plot_equity,
    positions,
    sharpe_difference,
    strategy_returns,
)
from brent_forecast.evaluation.statistics import (
    BootstrapOptions,
    accuracy,
    accuracy_vs_no_information,
    auc,
    auc_difference,
    bootstrap_interval,
    brier,
    brier_skill,
    delong_test,
    holm,
    plot_reliability,
)

logger = logging.getLogger(__name__)

CLIMATOLOGY_KEY = "majority"
"""Baseline used as the reference forecast of the Brier skill score."""

BUY_AND_HOLD = "Buy & hold (always long)"
"""Reference strategy of the backtest."""


def build_report(
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
    evaluation: EvaluationSettings,
    backtest: BacktestSettings,
    seed: int,
) -> dict[str, Any]:
    """Compute every statistic of the report as a JSON-serialisable dict."""
    y = predictions["label"].to_numpy(dtype=float)
    candidates: dict[str, dict[str, Any]] = metrics["models"]
    proba = {
        name: predictions[f"proba_{m['key']}"].to_numpy(dtype=float)
        for name, m in candidates.items()
    }
    boot: BootstrapOptions = {
        "block_length": evaluation.block_length,
        "n_resamples": evaluation.bootstrap_resamples,
        "confidence": evaluation.confidence_level,
        "seed": seed,
    }
    climatology = next(
        (proba[n] for n, m in candidates.items() if m["key"] == CLIMATOLOGY_KEY), None
    )

    per_candidate: dict[str, dict[str, Any]] = {}
    for name, m in candidates.items():
        p = proba[name]
        per_candidate[name] = {
            "kind": m["kind"],
            "auc": bootstrap_interval(auc, y, p, **boot).as_dict(),
            "accuracy": bootstrap_interval(accuracy, y, p, **boot).as_dict(),
            "accuracy_test": asdict(accuracy_vs_no_information(y, p)),
            "brier": brier(y, p),
            "brier_skill": None if climatology is None else brier_skill(y, p, climatology),
        }

    baselines = [n for n, m in candidates.items() if m["kind"] == "baseline"]
    models = [n for n, m in candidates.items() if m["kind"] == "model"]
    reference = max(baselines, key=lambda n: per_candidate[n]["auc"]["estimate"])
    comparisons: dict[str, dict[str, Any]] = {}
    for name in models:
        delta = bootstrap_interval(auc_difference, y, proba[name], proba[reference], **boot)
        comparisons[name] = {
            "delta_auc": delta.as_dict(),
            "delong": asdict(delong_test(y, proba[name], proba[reference])),
        }
    adjusted = holm({n: c["delong"]["p_value"] for n, c in comparisons.items()})
    for name, comparison in comparisons.items():
        comparison["p_holm"] = adjusted[name]
        comparison["significant"] = bool(
            adjusted[name] < evaluation.alpha and comparison["delta_auc"]["low"] > 0
        )

    trading = _backtest(predictions, candidates, backtest, boot)

    return {
        "protocol": metrics["protocol"],
        "data": metrics.get("data", {}),
        "n_days": len(y),
        "up_rate": float(y.mean()),
        "period": [str(predictions["date"].min().date()), str(predictions["date"].max().date())],
        "bootstrap": {**boot, "method": "circular block bootstrap"},
        "alpha": evaluation.alpha,
        "candidates": per_candidate,
        "reference_baseline": reference,
        "comparisons": comparisons,
        "backtest": {"settings": backtest.model_dump(), "strategies": trading},
        "conclusion": _conclusion(
            per_candidate, comparisons, trading, reference, evaluation.alpha, backtest
        ),
    }


def strategies(
    predictions: pd.DataFrame, candidates: dict[str, dict[str, Any]], backtest: BacktestSettings
) -> dict[str, pd.DataFrame]:
    """Daily positions and returns of buy & hold and of every candidate's strategy."""
    next_return = predictions["next_return"]
    out = {
        BUY_AND_HOLD: strategy_returns(
            pd.Series(1.0, index=predictions.index), next_return, backtest.cost_bps
        )
    }
    for name, m in candidates.items():
        position = positions(
            predictions[f"proba_{m['key']}"], backtest.threshold, allow_short=backtest.allow_short
        )
        out[name] = strategy_returns(position, next_return, backtest.cost_bps)
    return out


def _backtest(
    predictions: pd.DataFrame,
    candidates: dict[str, dict[str, Any]],
    backtest: BacktestSettings,
    boot: BootstrapOptions,
) -> dict[str, dict[str, Any]]:
    """Measure every strategy and its Sharpe difference against buy & hold."""
    runs = strategies(predictions, candidates, backtest)
    benchmark = runs[BUY_AND_HOLD]["net"].to_numpy()
    delta_sharpe = partial(sharpe_difference, periods_per_year=backtest.periods_per_year)
    out: dict[str, dict[str, Any]] = {}
    for name, returns in runs.items():
        is_reference = name == BUY_AND_HOLD
        delta = (
            None
            if is_reference
            else bootstrap_interval(
                delta_sharpe, returns["net"].to_numpy(), benchmark, **boot
            ).as_dict()
        )
        out[name] = {
            "kind": "reference" if is_reference else candidates[name]["kind"],
            **performance(returns, backtest.periods_per_year),
            "delta_sharpe": delta,
        }
    return out


def _conclusion(
    per_candidate: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    trading: dict[str, dict[str, Any]],
    reference: str,
    alpha: float,
    backtest: BacktestSettings,
) -> list[str]:
    """Plain-language findings derived only from the computed statistics."""
    models = list(comparisons)
    ref_auc = per_candidate[reference]["auc"]["estimate"]
    winners = [n for n in models if comparisons[n]["significant"]]
    above_chance = [n for n in models if per_candidate[n]["auc"]["low"] > 0.5]
    accurate = [n for n in models if per_candidate[n]["accuracy_test"]["p_value"] < alpha]
    calibrated = [n for n in models if (per_candidate[n]["brier_skill"] or 0.0) > 0]
    profitable = [n for n in models if trading[n]["delta_sharpe"]["low"] > 0]

    lines: list[str] = []
    if winners:
        lines.append(
            f"{_names(winners)} significantly outperform(s) the best naive baseline "
            f"({reference}, AUC {ref_auc:.3f}) at α = {alpha} after Holm correction, "
            "with a block-bootstrap interval of the AUC difference above zero."
        )
    else:
        lines.append(
            f"No model has a significantly higher out-of-sample AUC than the best naive "
            f"baseline ({reference}, AUC {ref_auc:.3f}) at α = {alpha} after Holm correction."
        )
    lines.append(
        f"Models whose AUC interval lies entirely above 0.5: {_names(above_chance)}."
        if above_chance
        else "Every model's AUC confidence interval contains 0.5 (chance level)."
    )
    lines.append(
        f"Accuracy significantly above the no-information rate: {_names(accurate)}."
        if accurate
        else "No model's accuracy is significantly above the no-information rate."
    )
    lines.append(
        f"Positive Brier skill against the climatological forecast: {_names(calibrated)}."
        if calibrated
        else "No model's probabilities beat the climatological forecast (Brier skill <= 0)."
    )
    costs = f"{backtest.cost_bps:g} bps per trade"
    lines.append(
        f"Sharpe ratio significantly above buy & hold after costs ({costs}): {_names(profitable)}."
        if profitable
        else f"No model's strategy has a significantly higher Sharpe ratio than buy & hold "
        f"after costs ({costs}); buy & hold Sharpe: {trading[BUY_AND_HOLD]['sharpe']:.2f}."
    )
    if not winners and not above_chance and not profitable:
        lines.append(
            "Conclusion: the evidence does not support next-day directional skill; "
            "the models should be treated as equivalent to a naive forecast."
        )
    return lines


def _names(names: list[str]) -> str:
    return ", ".join(names)


def _provenance(data: dict[str, Any]) -> str:
    matches = data.get("matches_recorded_checksums")
    if matches is True:
        return "Inputs match the recorded SHA-256 checksums of the Kaggle dataset (real data)."
    if matches is False:
        return "**Inputs do NOT match the recorded checksums**: the data was modified or replaced."
    return (
        "No checksums are recorded, so the provenance of the inputs is **unverified** "
        "(the numbers may come from synthetic data)."
    )


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as Markdown."""
    proto, boot = report["protocol"], report["bootstrap"]
    level = round(100 * boot["confidence"])
    out = [
        "# Statistical evaluation report",
        "",
        f"- Out-of-sample period: {report['period'][0]} to {report['period'][1]} "
        f"({report['n_days']} days, {100 * report['up_rate']:.1f} % up days).",
        f"- Walk-forward: {proto['n_windows']} windows of {proto['test_window']} days, "
        f"{proto['mode']} training window, purge {proto['purge']}, embargo {proto['embargo']}.",
        f"- Intervals: {level} % {boot['method']} (block {boot['block_length']} days, "
        f"{boot['n_resamples']} resamples, seed {boot['seed']}).",
        f"- Data: {_provenance(report['data'])}",
        "",
        "## Candidates",
        "",
        f"| Candidate | Kind | AUC [{level} % CI] | Accuracy [{level} % CI] | "
        "p (acc > NIR) | Brier | Brier skill |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, c in report["candidates"].items():
        skill = "—" if c["brier_skill"] is None else f"{c['brier_skill']:+.4f}"
        out.append(
            f"| {name} | {c['kind']} | {_ci(c['auc'])} | {_ci(c['accuracy'])} | "
            f"{c['accuracy_test']['p_value']:.3f} | {c['brier']:.4f} | {skill} |"
        )
    out += [
        "",
        f"NIR (no-information rate) = {_nir(report):.4f}. "
        "Brier skill is relative to the majority-class forecast.",
        "",
        f"## Models vs the reference baseline ({report['reference_baseline']})",
        "",
        f"| Model | ΔAUC [{level} % CI] | DeLong z | p | p (Holm) | Significant |",
        "|---|---|---|---|---|---|",
    ]
    for name, c in report["comparisons"].items():
        out.append(
            f"| {name} | {_ci(c['delta_auc'], signed=True)} | {c['delong']['z']:.2f} | "
            f"{c['delong']['p_value']:.3f} | {c['p_holm']:.3f} | "
            f"{'yes' if c['significant'] else 'no'} |"
        )
    bt = report["backtest"]["settings"]
    short = "long/short" if bt["allow_short"] else "long/flat"
    out += [
        "",
        "## Economic backtest",
        "",
        f"{short.capitalize()} strategy: long when P(up) >= {bt['threshold']}, "
        f"costs {bt['cost_bps']:g} bps per unit of turnover, zero risk-free rate.",
        "",
        f"| Strategy | Kind | CAGR | Volatility | Sharpe | ΔSharpe vs B&H [{level} % CI] | "
        "Max drawdown | Hit ratio | Exposure | Trades |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, st in report["backtest"]["strategies"].items():
        delta = (
            "—" if st["delta_sharpe"] is None else _ci(st["delta_sharpe"], signed=True, digits=2)
        )
        hit = "—" if st["hit_ratio"] is None else f"{st['hit_ratio']:.3f}"
        out.append(
            f"| {name} | {st['kind']} | {st['cagr']:+.2%} | {st['volatility']:.2%} | "
            f"{st['sharpe']:.2f} | {delta} | {st['max_drawdown']:.2%} | {hit} | "
            f"{st['exposure']:.0%} | {st['n_trades']:.0f} |"
        )
    out += ["", "## Conclusion", ""]
    out += [f"- {line}" for line in report["conclusion"]]
    out += [
        "",
        "DeLong and binomial tests assume independent days; the block-bootstrap intervals "
        "do not, and a comparison is called significant only if both agree.",
        "",
    ]
    return "\n".join(out)


def _ci(interval: dict[str, float], *, signed: bool = False, digits: int = 4) -> str:
    fmt = f"{'+' if signed else ''}.{digits}f"
    return f"{interval['estimate']:{fmt}} [{interval['low']:{fmt}}, {interval['high']:{fmt}}]"


def _nir(report: dict[str, Any]) -> float:
    first = next(iter(report["candidates"].values()))
    return float(first["accuracy_test"]["no_information_rate"])


def generate_report(settings: Settings) -> Path:
    """Build the report from the last training run and write it; return the Markdown path.

    Raises
    ------
    FileNotFoundError
        If ``brent train`` has not produced predictions and metrics yet.
    """
    paths = settings.paths
    for path in (paths.predictions_file, paths.metrics_file):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run `brent train` first.")
    predictions = pd.read_csv(paths.predictions_file, parse_dates=["date"])
    metrics = json.loads(paths.metrics_file.read_text(encoding="utf-8"))

    report = build_report(
        predictions, metrics, settings.evaluation, settings.backtest, settings.seed
    )
    models = {n: m for n, m in metrics["models"].items() if m["kind"] == "model"}
    runs = strategies(predictions, models, settings.backtest)
    plot_equity(predictions["date"], {n: r["net"] for n, r in runs.items()}, paths.plots_dir)
    plot_reliability(
        predictions["label"],
        {n: predictions[f"proba_{m['key']}"] for n, m in models.items()},
        settings.evaluation.reliability_bins,
        paths.plots_dir,
    )
    paths.report_data_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    paths.report_file.write_text(render_markdown(report), encoding="utf-8")
    for line in report["conclusion"]:
        logger.info("%s", line)
    return paths.report_file
