"""Static checks of dvc.yaml: every stage must stay in sync with the CLI and the config.

They run without DVC or data, so a renamed parameter, command or module breaks
CI instead of breaking `dvc repro` later.
"""

import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from brent_forecast.cli import app
from conftest import CONFIG_PATH, REPO_ROOT

DVC_FILE = REPO_ROOT / "dvc.yaml"
GENERATED = ("data/", "results/")
STAGES: dict[str, Any] = yaml.safe_load(DVC_FILE.read_text())["stages"]


def _paths(entries: list[Any]) -> list[str]:
    return [next(iter(e)) if isinstance(e, dict) else e for e in entries]


def _outputs(stage: dict[str, Any]) -> list[str]:
    return [p for kind in ("outs", "metrics", "plots") for p in _paths(stage.get(kind, []))]


def test_stages_form_the_documented_chain() -> None:
    assert list(STAGES) == ["download", "validate", "featurize", "train", "evaluate", "explain"]


@pytest.mark.parametrize("name", list(STAGES))
def test_every_command_is_a_valid_cli_call(name: str) -> None:
    args = shlex.split(STAGES[name]["cmd"])
    assert args[0] == "brent"

    result = CliRunner().invoke(app, [*args[1:], "--help"])

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("name", list(STAGES))
def test_every_parameter_exists_in_the_config(name: str) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    for group in STAGES[name].get("params", []):
        for key in group[str(CONFIG_PATH.relative_to(REPO_ROOT))]:
            node = config
            for part in key.split("."):
                assert part in node, f"{name}: {key} not in {CONFIG_PATH.name}"
                node = node[part]


@pytest.mark.parametrize("name", list(STAGES))
def test_source_dependencies_exist(name: str) -> None:
    for dep in STAGES[name].get("deps", []):
        if not dep.startswith(GENERATED):
            assert (REPO_ROOT / dep).exists(), f"{name}: missing dependency {dep}"


def test_outputs_do_not_overlap_and_feed_later_stages() -> None:
    owners: dict[str, str] = {}
    for name, stage in STAGES.items():
        for out in _outputs(stage):
            for other, owner in owners.items():
                assert not (out.startswith(other) or other.startswith(out)), (out, owner)
            owners[out] = name
        for dep in stage.get("deps", []):
            if dep.startswith(GENERATED):
                assert dep in owners, f"{name} depends on {dep}, which no earlier stage writes"


def test_dates_in_the_config_are_strings_for_dvc() -> None:
    """DVC stores parameters in dvc.lock as JSON: YAML dates must be quoted."""
    text = Path(CONFIG_PATH).read_text()
    config = yaml.safe_load(text)

    def scalars(node: Any) -> list[Any]:
        if isinstance(node, dict):
            return [v for value in node.values() for v in scalars(value)]
        if isinstance(node, list):
            return [v for value in node for v in scalars(value)]
        return [node]

    assert all(isinstance(v, str | int | float | bool) or v is None for v in scalars(config))
