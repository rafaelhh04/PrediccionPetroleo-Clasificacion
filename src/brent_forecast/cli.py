"""Command-line interface for the Brent direction classifier."""

import logging
from pathlib import Path

import typer

from brent_forecast.data.download import CHECKSUMS_PATH, DATA_DIR

app = typer.Typer(
    name="brent",
    help="Next-day Brent crude oil price direction classifier.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Acquire and verify the raw datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """Next-day Brent crude oil price direction classifier."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def train() -> None:
    """Run the full pipeline: load, preprocess, tune, train and evaluate on test."""
    from brent_forecast.pipeline import run

    run()


@data_app.command("download")
def data_download(
    data_dir: Path = typer.Option(DATA_DIR, help="Destination directory for the CSV files."),
    checksums: Path = typer.Option(CHECKSUMS_PATH, help="JSON file with SHA-256 digests."),
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

    try:
        files = download_dataset(
            data_dir, checksums, force=force, update_checksums=update_checksums
        )
    except (DataDownloadError, ChecksumMismatchError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    for path in files.values():
        typer.echo(f"Ready: {path}")


@data_app.command("verify")
def data_verify(
    data_dir: Path = typer.Option(DATA_DIR, help="Directory containing the CSV files."),
    checksums: Path = typer.Option(CHECKSUMS_PATH, help="JSON file with SHA-256 digests."),
) -> None:
    """Verify the SHA-256 checksums of the local data files."""
    from brent_forecast.data.download import ChecksumMismatchError, verify_checksums

    try:
        verify_checksums(data_dir, checksums)
    except (FileNotFoundError, ChecksumMismatchError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("All data files verified.")
