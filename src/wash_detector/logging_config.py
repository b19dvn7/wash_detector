"""Logging configuration for wash_detector.

Provides centralized logging setup with sensible defaults.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(
    level: int = logging.WARNING,
    *,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure logging for the wash_detector package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to write logs to file
        format_string: Custom format string (uses sensible default if None)
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Configure root logger for wash_detector namespace
    logger = logging.getLogger("wash_detector")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler (stderr to not mix with CLI output on stdout)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the wash_detector namespace.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    if not name.startswith("wash_detector"):
        name = f"wash_detector.{name}"
    return logging.getLogger(name)
