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
    """Build a complete ML-ready dataset: original OHLC + features + label.

    Combines the original OHLC columns, build_features' engineered
    columns, and create_labels' label column into a single dataset, then
    drops any row with a NaN in an engineered feature or the label
    (indicator warm-up at the start, label horizon at the end) -- NOT a
    NaN in a passthrough OHLC column, since some providers legitimately
    report no volume for certain symbols (XAU/USD, say), and that should
    not disqualify an otherwise-usable row. Raises ValueError if fewer than
    `min_rows` remain.

    The original OHLC columns are kept, not just the engineered features,
    because downstream trading evaluation needs real close prices to
    compute realized returns -- separate_features_and_target (in
    ml/dataset.py) already handles selecting only FEATURE_COLUMNS out of
    whatever this returns, so carrying extra columns through here is safe
    and doesn't leak into the model's inputs.

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

    dataset = dataframe.copy()
    dataset[features.columns] = features
    dataset["label"] = labels

    required_columns = list(features.columns) + ["label"]
    dataset = dataset.dropna(subset=required_columns)

    if len(dataset) < min_rows:
        raise ValueError(
            f"Only {len(dataset)} usable rows after dropping NaNs; "
            f"minimum required is {min_rows}",
        )

    return dataset