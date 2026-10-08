"""Tests for brent_forecast.evaluation.report (``brent report``)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from brent_forecast.config import Settings
from brent_forecast.evaluation.report import build_report, generate_report, render_markdown

PROTOCOL = {"n_windows": 4, "test_window": 63, "mode": "expanding", "purge": 1, "embargo": 5}


def _run_artefacts(signal: float, n: int = 252, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """predictions.csv / metrics.json of a fake run with one model and two baselines."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.52).astype(float)
    model = 1 / (1 + np.exp(-(signal * (2 * y - 1) + rng.normal(size=n))))
    lag = np.r_[y[0], y[:-1]]  # persistence: yesterday's label
    predictions = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "label": y,
            "window": np.arange(n) // 63,
            "proba_model": model,
            "proba_majority": np.full(n, 0.52),
            "proba_persistence": lag,
        }
    )
    metrics = {
        "protocol": PROTOCOL,
        "data": {"sha256": {}, "matches_recorded_checksums": None},
        "models": {
            "Model": {"key": "model", "kind": "model"},
            "Majority class": {"key": "majority", "kind": "baseline"},
            "Persistence": {"key": "persistence", "kind": "baseline"},
        },
    }
    return predictions, metrics


def _report(settings: Settings, signal: float) -> dict[str, Any]:
    predictions, metrics = _run_artefacts(signal)
    return build_report(predictions, metrics, settings.evaluation, seed=0)


def test_a_real_signal_is_significant(fast_settings: Settings) -> None:
    report = _report(fast_settings, signal=1.0)

    comparison = report["comparisons"]["Model"]
    assert report["reference_baseline"] in {"Majority class", "Persistence"}
    assert comparison["significant"]
    assert comparison["delta_auc"]["low"] > 0
    assert comparison["p_holm"] < 0.05
    assert report["candidates"]["Model"]["brier_skill"] > 0
    assert "significantly outperform" in report["conclusion"][0]
    assert not any(line.startswith("Conclusion:") for line in report["conclusion"])


def test_noise_is_reported_honestly(fast_settings: Settings) -> None:
    report = _report(fast_settings, signal=0.0)

    assert not report["comparisons"]["Model"]["significant"]
    assert report["conclusion"][0].startswith("No model has a significantly higher")
    assert report["conclusion"][-1].startswith("Conclusion: the evidence does not support")


def test_report_structure(fast_settings: Settings) -> None:
    report = _report(fast_settings, signal=0.5)

    assert set(report["candidates"]) == {"Model", "Majority class", "Persistence"}
    assert set(report["comparisons"]) == {"Model"}  # baselines are not compared
    majority = report["candidates"]["Majority class"]
    assert majority["auc"]["estimate"] == 0.5
    assert majority["brier_skill"] == 0.0
    assert report["n_days"] == 252
    assert report["bootstrap"]["n_resamples"] == 200
    json.dumps(report)  # serialisable


def test_brier_skill_is_omitted_without_a_majority_baseline(fast_settings: Settings) -> None:
    predictions, metrics = _run_artefacts(0.5)
    del metrics["models"]["Majority class"]

    report = build_report(predictions, metrics, fast_settings.evaluation, seed=0)

    assert report["candidates"]["Model"]["brier_skill"] is None
    assert "| — |" in render_markdown(report)


@pytest.mark.parametrize(
    ("matches", "text"),
    [(True, "real data"), (False, "do NOT match"), (None, "unverified")],
)
def test_markdown_states_the_data_provenance(
    fast_settings: Settings, matches: bool | None, text: str
) -> None:
    report = _report(fast_settings, signal=0.5)
    report["data"] = {"matches_recorded_checksums": matches}

    markdown = render_markdown(report)

    assert text in markdown
    assert "## Models vs the reference baseline" in markdown
    assert "| Model | model |" in markdown


def test_generate_report_requires_a_training_run(fast_settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="brent train"):
        generate_report(fast_settings)


def test_generate_report_writes_markdown_json_and_plot(fast_settings: Settings) -> None:
    paths = fast_settings.paths
    paths.results_dir.mkdir(parents=True)
    predictions, metrics = _run_artefacts(0.5)
    predictions.to_csv(paths.predictions_file, index=False)
    paths.metrics_file.write_text(json.dumps(metrics))

    path = generate_report(fast_settings)

    assert path == paths.report_file
    assert path.read_text().startswith("# Statistical evaluation report")
    assert json.loads(paths.report_data_file.read_text())["reference_baseline"]
    assert (paths.plots_dir / "reliability_diagram.png").is_file()


def test_report_is_deterministic(fast_settings: Settings, tmp_path: Path) -> None:
    assert _report(fast_settings, 0.3) == _report(fast_settings, 0.3)
