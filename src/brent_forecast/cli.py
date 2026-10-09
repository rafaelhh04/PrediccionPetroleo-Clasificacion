"""Command-line interface for the Brent direction classifier (``brent``)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from brent_forecast.config import DEFAULT_CONFIG_PATH, Settings, load_settings
from brent_forecast.logging_config import LOG_LEVELS, setup_logging

app = typer.Typer(
    name="brent",
    help="Next-day Brent crude oil price direction classifier.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Acquire and verify the raw datasets.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect the resolved configuration.", no_args_is_help=True)
registry_app = typer.Typer(help="Inspect the MLflow model registry.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")
app.add_typer(registry_app, name="registry")


@dataclass(frozen=True)
class _State:
    """Global options shared by every command."""

    config_path: Path
    log_level: str


def _validate_log_level(value: str) -> str:
    level = value.upper()
    if level not in LOG_LEVELS:
        raise typer.BadParameter(f"choose from {', '.join(LOG_LEVELS)}")
    return level


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="YAML configuration file (values can be overridden with BRENT_* env vars).",
        ),
    ] = DEFAULT_CONFIG_PATH,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            "-l",
            callback=_validate_log_level,
            help=f"Logging level: {', '.join(LOG_LEVELS)}.",
        ),
    ] = "INFO",
) -> None:
    """Next-day Brent crude oil price direction classifier."""
    setup_logging(log_level)
    ctx.obj = _State(config_path=config, log_level=log_level)


def _state(ctx: typer.Context) -> _State:
    state = ctx.obj
    assert isinstance(state, _State)
    return state


def _settings(ctx: typer.Context) -> Settings:
    """Load the settings for the config path stored by the root callback."""
    try:
        return load_settings(_state(ctx).config_path)
    except (FileNotFoundError, ValidationError) as exc:
        typer.secho(f"Invalid configuration: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def train(ctx: typer.Context) -> None:
    """Run the full pipeline: load, preprocess, tune, train and evaluate on test."""
    from brent_forecast.pipeline import run

    settings = _settings(ctx)
    setup_logging(_state(ctx).log_level, log_file=settings.paths.log_file)
    result = run(settings)
    if settings.tracking.enabled:
        from brent_forecast.tracking import log_training

        run_id = log_training(settings, result)
        typer.echo(f"MLflow run {run_id} ({settings.tracking.uri})")
    typer.echo(f"Log written to {settings.paths.log_file}")


@app.command()
def report(ctx: typer.Context) -> None:
    """Build the statistical report (CIs, DeLong, binomial, calibration) of the last run."""
    from brent_forecast.evaluation.report import generate_report

    settings = _settings(ctx)
    try:
        path = generate_report(settings)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Report written to {path}")
    if settings.tracking.enabled:
        from brent_forecast.tracking import log_evaluation

        try:
            decision = log_evaluation(settings)
        except FileNotFoundError as exc:
            typer.secho(str(exc), fg=typer.colors.YELLOW, err=True)
            return
        typer.echo(f"champion = {decision.champion}, challenger = {decision.challenger}")
        typer.echo(decision.reason)


@app.command()
def explain(ctx: typer.Context) -> None:
    """Explain the trained models: permutation importance and SHAP (global and local)."""
    from brent_forecast.evaluation.explain import generate_explanations

    settings = _settings(ctx)
    try:
        path = generate_explanations(settings)
    except (FileNotFoundError, KeyError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Explanations written to {path}")


@registry_app.command("show")
def registry_show(ctx: typer.Context) -> None:
    """Print the aliases of the registered model with their versions and tags."""
    from brent_forecast.tracking import registry_aliases

    settings = _settings(ctx)
    aliases = registry_aliases(settings)
    if not aliases:
        typer.echo(f"No model registered as {settings.tracking.registered_model!r} yet.")
        return
    typer.echo(json.dumps(aliases, indent=2))


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Print the resolved configuration (YAML + environment overrides) as JSON."""
    typer.echo(json.dumps(_settings(ctx).model_dump(mode="json"), indent=2))


@data_app.command("download")
def data_download(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option("--force", help="Ignore the kagglehub cache.")] = False,
    update_checksums: Annotated[
        bool,
        typer.Option(
            "--update-checksums", help="Overwrite recorded digests with the downloaded ones."
        ),
    ] = False,
) -> None:
    """Download the Kaggle dataset into the data directory and verify checksums."""
    from brent_forecast.data.download import (
        ChecksumMismatchError,
        DataDownloadError,
        download_dataset,
    )

    settings = _settings(ctx)
    try:
        files = download_dataset(
            settings.paths.data_dir,
            settings.paths.checksums_file,
            settings.data,
            force=force,
            update_checksums=update_checksums,
        )
    except (DataDownloadError, ChecksumMismatchError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    for path in files.values():
        typer.echo(f"Ready: {path}")


@data_app.command("verify")
def data_verify(ctx: typer.Context) -> None:
    """Verify the SHA-256 checksums of the local data files."""
    from brent_forecast.data.download import ChecksumMismatchError, verify_checksums

    settings = _settings(ctx)
    try:
        verify_checksums(settings.paths.data_dir, settings.paths.checksums_file, settings.data)
    except (FileNotFoundError, ChecksumMismatchError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("All data files verified.")
