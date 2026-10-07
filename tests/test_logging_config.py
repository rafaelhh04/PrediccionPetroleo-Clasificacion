"""Tests for brent_forecast.logging_config."""

import logging
from pathlib import Path

import pytest

from brent_forecast.logging_config import setup_logging


def test_console_only() -> None:
    setup_logging("debug")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


def test_file_handler_writes_and_overwrites(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "run.log"
    log_file.parent.mkdir()
    log_file.write_text("previous run\n")

    setup_logging("INFO", log_file=log_file)
    logging.getLogger("brent_forecast.test").info("hello %s", "world")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "previous run" not in content
    assert "INFO    brent_forecast.test: hello world" in content


def test_calling_twice_replaces_handlers(tmp_path: Path) -> None:
    setup_logging("INFO", log_file=tmp_path / "a.log")
    setup_logging("WARNING")

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.WARNING


def test_noisy_loggers_are_quietened() -> None:
    setup_logging("DEBUG")

    assert logging.getLogger("matplotlib").level == logging.WARNING


def test_numeric_level_is_accepted() -> None:
    setup_logging(logging.ERROR)

    assert logging.getLogger().level == logging.ERROR


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        setup_logging("VERBOSE")
