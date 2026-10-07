"""Logging configuration shared by every entry point.

Library modules only create loggers with ``logging.getLogger(__name__)``;
handlers are configured exclusively here, by the application (the CLI).
"""

import logging
import sys
from pathlib import Path

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Third-party loggers that are too chatty at DEBUG level.
_NOISY_LOGGERS = ("matplotlib", "PIL", "urllib3", "filelock")


def setup_logging(level: str | int = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger with a console handler and an optional file handler.

    Calling it again replaces the previous handlers, so it is safe to call once
    for console output and later again to add the run's log file.

    Parameters
    ----------
    level
        Minimum level for the project's loggers (name or numeric value).
    log_file
        If given, the log is also written to this file (overwritten per run).
    """
    if isinstance(level, str):
        level = level.upper()
        if level not in LOG_LEVELS:
            raise ValueError(f"Unknown log level {level!r}; choose from {LOG_LEVELS}")

    # The Windows console defaults to cp1252; avoid UnicodeEncodeError on
    # characters such as "→" by replacing anything it cannot represent.
    stream = sys.stderr
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(stream)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATEFMT))
    handlers.append(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        handlers.append(file_handler)

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
    logging.captureWarnings(True)
