"""Logging configuration shared by every entry point.

Library modules only create loggers with ``logging.getLogger(__name__)``;
handlers are configured exclusively here, by the application (the CLI).

Two console formats: ``text`` (human-readable, the default) and ``json`` (one
object per line, for log collectors). JSON records carry the request id of the
HTTP request being served (:data:`request_id`, set by the API middleware) and
any ``extra`` fields passed to the logger.
"""

import contextlib
import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_FORMATS = ("text", "json")
LogFormat = Literal["text", "json"]

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
"""Id of the HTTP request handled by the current context (``None`` outside requests)."""

# Attributes every LogRecord has; anything else was passed with ``extra=``.
_RECORD_ATTRS = frozenset(
    set(vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None))) | {"message", "asctime"}
)

CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

# Third-party loggers that are too chatty at DEBUG level.
_NOISY_LOGGERS = ("matplotlib", "PIL", "urllib3", "filelock")


class RequestIdFilter(logging.Filter):
    """Attach the current request id (or ``None``) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add ``record.request_id``; never drop a record."""
        record.request_id = request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per record: time, level, logger, message, request id and extras."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` (non-JSON extras are converted with ``str``)."""
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RECORD_ATTRS and key != "request_id":
                payload[key] = value
        rid = getattr(record, "request_id", None)
        if rid is not None:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(
    level: str | int = "INFO", log_file: Path | None = None, fmt: LogFormat = "text"
) -> None:
    """Configure the root logger with a console handler and an optional file handler.

    Calling it again replaces the previous handlers, so it is safe to call once
    for console output and later again to add the run's log file.

    Parameters
    ----------
    level
        Minimum level for the project's loggers (name or numeric value).
    log_file
        If given, the log is also written to this file (overwritten per run).
    fmt
        Console format: ``text`` or ``json`` (the log file is always text).
    """
    if fmt not in LOG_FORMATS:
        raise ValueError(f"Unknown log format {fmt!r}; choose from {LOG_FORMATS}")
    if isinstance(level, str):
        level = level.upper()
        if level not in LOG_LEVELS:
            raise ValueError(f"Unknown log level {level!r}; choose from {LOG_LEVELS}")

    # The Windows console defaults to cp1252; avoid UnicodeEncodeError on
    # characters such as "→" by replacing anything it cannot represent.
    stream = sys.stderr
    if hasattr(stream, "reconfigure"):
        with contextlib.suppress(OSError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(stream)
    if fmt == "json":
        console.setFormatter(JsonFormatter())
    else:
        console.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATEFMT))
    console.addFilter(RequestIdFilter())
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
