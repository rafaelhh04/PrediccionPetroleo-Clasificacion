"""Streamlit entry point: ``brent dashboard`` runs ``streamlit run`` on this file.

The configuration file comes after ``--`` (``streamlit run app.py -- --config PATH``);
``BRENT_*`` variables apply as everywhere else.
"""

import argparse
from pathlib import Path

from brent_forecast.config import DEFAULT_CONFIG_PATH, load_settings
from brent_forecast.dashboard.data import ApiClient
from brent_forecast.dashboard.page import render


def main(argv: list[str] | None = None) -> None:
    """Load the settings and draw the page."""
    parser = argparse.ArgumentParser(prog="brent-dashboard")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args, _ = parser.parse_known_args(argv)
    settings = load_settings(Path(args.config))
    render(settings, ApiClient.from_settings(settings))


main()
