"""Command-line interface for the Brent direction classifier."""

import typer

app = typer.Typer(
    name="brent",
    help="Next-day Brent crude oil price direction classifier.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Next-day Brent crude oil price direction classifier."""


@app.command()
def train() -> None:
    """Run the full pipeline: load, preprocess, tune, train and evaluate on test."""
    from brent_forecast.pipeline import run

    run()
