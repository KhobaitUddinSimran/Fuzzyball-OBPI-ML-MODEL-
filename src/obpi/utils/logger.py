"""Structured logging setup for OBPI."""

import logging
import sys
from typing import Any

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: int = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    stream: Any = sys.stderr,
) -> None:
    """Configure root logger for OBPI.

    Args:
        level: Logging level (default ``logging.INFO``).
        fmt: Log record format string.
        stream: Output stream (default ``sys.stderr``).
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("obpi")
    root.setLevel(level)
    root.handlers = [handler]
