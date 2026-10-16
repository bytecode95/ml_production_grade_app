
"""Centralized application logging with colored console output."""

import logging
import os
import sys
from logging.config import dictConfig

from colorama import Fore, Style, just_fix_windows_console

LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.RED + Style.BRIGHT,
}


class ColoredFormatter(logging.Formatter):
    """Apply colors to log levels while preserving readable output."""

    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS.get(record.levelno, "")
        message = super().format(record)

        if not color:
            return message

        return f"{color}{message}{Style.RESET_ALL}"


def configure_logging() -> None:
    """Configure application-wide logging once."""

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    if log_level not in {
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    }:
        raise ValueError(f"Invalid LOG_LEVEL: {log_level}")

    just_fix_windows_console()

    formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate output if configuration is called more than once.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
