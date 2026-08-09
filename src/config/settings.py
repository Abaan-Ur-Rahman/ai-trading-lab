"""
Configuration settings for AI Trading Lab.

This module defines the default application settings used throughout
the project. Settings represent configurable values that may differ
between environments or user preferences, while constants remain
fixed across the application.
"""

from dataclasses import dataclass

from config.constants import (
    DEFAULT_RANDOM_SEED,
    SUPPORTED_ASSETS,
    SUPPORTED_TIMEFRAMES,
)


@dataclass(slots=True)
class Settings:
    """
    Centralized application settings.

    An instance of this class stores the default configuration used
    throughout AI Trading Lab. Grouping related settings together
    improves maintainability and avoids scattered global variables.
    """

    default_asset: str = SUPPORTED_ASSETS[0]
    default_timeframe: str = SUPPORTED_TIMEFRAMES[4]
    log_level: str = "INFO"
    random_seed: int = DEFAULT_RANDOM_SEED