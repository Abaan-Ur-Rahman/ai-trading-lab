"""Build ML-ready feature columns from OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.technical_indicators import TechnicalIndicators

def multi_period_log_return(close: pd.Series, periods: int) -> pd.Series:
    """Calculate the log return of close over a configurable number of periods.

    Compares close[t] to close[t - periods], using only the current and past
    closing prices, so no future information enters the calculation. The
    first `periods` rows, where no sufficiently old price exists, are NaN.
    """
    if periods <= 0:
        raise ValueError("periods must be a positive integer")

    return np.log(close / close.shift(periods))

def build_features(
    dataframe: pd.DataFrame,
    indicators: TechnicalIndicators | None = None,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_length: int = 14,
    atr_length: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Build ML feature columns from OHLCV data.

    Does not mutate the input dataframe. Returns a new DataFrame with the
    same index as the input, containing exactly these columns: log_return,
    ema_gap_pct, rsi, atr_pct, macd_hist_pct.

    Leading rows without enough history for a given indicator contain NaN.
    No imputation is performed here; that is the pipeline's responsibility.
    """
    required_columns = {"high", "low", "close"}
    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(f"DataFrame must contain columns: {sorted(missing)}")

    indicators = indicators or TechnicalIndicators()

    close = dataframe["close"]

    log_return = np.log(close / close.shift(1))

    return_3 = multi_period_log_return(close, periods=3)
    return_10 = multi_period_log_return(close, periods=10)
    return_20 = multi_period_log_return(close, periods=20)

    ema_fast_series = indicators.exponential_moving_average(dataframe, length=ema_fast)
    ema_slow_series = indicators.exponential_moving_average(dataframe, length=ema_slow)
    ema_gap_pct = (ema_fast_series - ema_slow_series) / ema_slow_series

    rsi = indicators.relative_strength_index(dataframe, length=rsi_length)

    atr = indicators.average_true_range(dataframe, length=atr_length)
    atr_pct = atr / close

    macd_df = indicators.macd(dataframe, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    macd_hist_pct = macd_df[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"] / close

    return pd.DataFrame(
        {
            "log_return": log_return,
            "return_3": return_3,
            "return_10": return_10,
            "return_20": return_20,
            "ema_gap_pct": ema_gap_pct,
            "rsi": rsi,
            "atr_pct": atr_pct,
            "macd_hist_pct": macd_hist_pct,
        },
        index=dataframe.index,
    )