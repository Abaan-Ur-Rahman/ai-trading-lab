"""
Logging utilities for AI Trading Lab.

This module provides a centralized way to create and configure loggers.
Every module in the project should obtain its logger through
`get_logger()` instead of configuring logging independently.
"""

from __future__ import annotations

import logging

from config.paths import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Logs are written to both the console and the central application
    log file. The logs directory is created automatically if it does
    not exist.

    Parameters
    ----------
    name : str
        Usually the module name (__name__).

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "ai_trading_lab.log"

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger