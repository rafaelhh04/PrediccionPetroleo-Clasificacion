"""Command-line interface for the Brent direction classifier."""

import json
import logging
from pathlib import Path

import typer
from pydantic import ValidationError

from brent_forecast.config import DEFAULT_CONFIG_PATH, Settings, load_settings

app = typer.Typer(
    name="brent",
    help="Next-day Brent crude oil price direction classifier.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Acquire and verify the raw datasets.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect the resolved configuration.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="YAML configuration file (values can be overridden with BRENT_* env vars).",
    ),
) -> None:
    """Next-day Brent crude oil price direction classifier."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctx.obj = config


def _settings(ctx: typer.Context) -> Settings:
    """Load the settings for the config path stored by the root callback."""
    try:
        return load_settings(ctx.obj)
    except (FileNotFoundError, ValidationError) as exc:
        typer.secho(f"Invalid configuration: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def train(ctx: typer.Context) -> None:
    """Run the full pipeline: load, preprocess, tune, train and evaluate on test."""
    from brent_forecast.pipeline import run

    run(_settings(ctx))


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Print the resolved configuration (YAML + environment overrides) as JSON."""
    typer.echo(json.dumps(_settings(ctx).model_dump(mode="json"), indent=2))


@data_app.command("download")
def data_download(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", help="Ignore the kagglehub cache."),
    update_checksums: bool = typer.Option(
        False, "--update-checksums", help="Overwrite recorded digests with the downloaded ones."
    ),
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
