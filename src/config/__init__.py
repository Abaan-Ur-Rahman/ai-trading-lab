"""
Configuration package for AI Trading Lab.

This package centralizes project configuration, constants,
filesystem paths, and application settings.
"""

from .constants import (
    DEFAULT_RANDOM_SEED,
    PROJECT_NAME,
    PROJECT_VERSION,
    SUPPORTED_ASSETS,
    SUPPORTED_TIMEFRAMES,
)

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    DOCS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    REPORTS_DIR,
    RESEARCH_DIR,
    SRC_DIR,
    TESTS_DIR,
)

from .settings import Settings

__all__ = [
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "DEFAULT_RANDOM_SEED",
    "SUPPORTED_ASSETS",
    "SUPPORTED_TIMEFRAMES",
    "PROJECT_ROOT",
    "DATA_DIR",
    "DOCS_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "NOTEBOOKS_DIR",
    "REPORTS_DIR",
    "RESEARCH_DIR",
    "SRC_DIR",
    "TESTS_DIR",
    "Settings",
]