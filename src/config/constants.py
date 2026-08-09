"""
Project-wide constants for AI Trading Lab.

This module contains values that are considered constant across the
application. Unlike user-configurable settings, these values are not
expected to change during normal operation.
"""

# ---------------------------------------------------------------------
# Project Information
# ---------------------------------------------------------------------

PROJECT_NAME = "AI Trading Lab"
PROJECT_VERSION = "0.1.0-alpha"

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

DEFAULT_RANDOM_SEED = 42

# ---------------------------------------------------------------------
# Supported Assets (Version 1)
# ---------------------------------------------------------------------

SUPPORTED_ASSETS = (
    "XAUUSD",
    "XAGUSD",
)

# ---------------------------------------------------------------------
# Supported Timeframes
# ---------------------------------------------------------------------

SUPPORTED_TIMEFRAMES = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
)