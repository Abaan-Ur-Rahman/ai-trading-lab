"""Combine feature building and labeling into a single ML-ready dataset."""

from __future__ import annotations

import pandas as pd

from features.feature_builder import build_features
from features.labeling import create_labels
from features.technical_indicators import TechnicalIndicators


def build_feature_dataset(
    dataframe: pd.DataFrame,
    indicators: TechnicalIndicators | None = None,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_length: int = 14,
    atr_length: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    horizon: int = 5,
    threshold: float = 0.005,
    min_rows: int = 100,
) -> pd.DataFrame:
    """Build a complete ML-ready dataset: features + label, with zero NaNs.

    Combines build_features and create_labels, drops any row containing a
    NaN in any column (indicator warm-up at the start, label horizon at
    the end), and raises ValueError if fewer than `min_rows` remain.

    Does not mutate the input dataframe.
    """
    if min_rows <= 0:
        raise ValueError("min_rows must be greater than 0")

    indicators = indicators or TechnicalIndicators()

    features = build_features(
        dataframe,
        indicators=indicators,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_length=rsi_length,
        atr_length=atr_length,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    )

    labels = create_labels(dataframe, horizon=horizon, threshold=threshold)

    dataset = features.copy()
    dataset["label"] = labels
    dataset = dataset.dropna()

    if len(dataset) < min_rows:
        raise ValueError(
            f"Only {len(dataset)} usable rows after dropping NaNs; "
            f"minimum required is {min_rows}",
        )

    return dataset