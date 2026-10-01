"""Live-style single-point inference for the ML layer.

This is the only place in src/ml/ that is allowed to call into
src/features/ for feature computation on fresh, unlabeled data. Training
and evaluation always work with an already-built dataset (via
build_feature_dataset); inference is the one path that takes raw OHLCV
"as of now" and has to build features itself, since there is no future
data yet to build a dataset from.
"""

from __future__ import annotations

import pandas as pd

from features.pipeline import build_features
from features.scaling import FeatureScaler, apply_scaler
from features.technical_indicators import TechnicalIndicators
from ml.dataset import FEATURE_COLUMNS
from ml.models.base import ModelWrapper


def predict_from_ohlcv(
    ohlcv: pd.DataFrame,
    model: ModelWrapper,
    scaler: FeatureScaler,
    indicators: TechnicalIndicators | None = None,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_length: int = 14,
    atr_length: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.Series:
    """Predict SELL/HOLD/BUY probabilities for the most recent row of `ohlcv`.

    `ohlcv` should be data up to and including "now" -- the most recent
    row is treated as the point to predict for. This function builds
    features over the full history (needed for the trailing windows
    indicators require), but only ever predicts on the single latest row.

    Does not mutate `ohlcv`.

    Args:
        ohlcv: Historical OHLCV data, chronologically ordered, ending at
            the point you want a prediction for.
        model: A fitted ModelWrapper.
        scaler: The FeatureScaler fitted during that model's training.
        indicators: Optional TechnicalIndicators instance (for test
            injection); a default instance is used if not provided.
        ema_fast, ema_slow, rsi_length, atr_length, macd_fast, macd_slow,
            macd_signal: Must match the values used when the model's
            training dataset was built, or the computed features will not
            mean what the model was trained on.

    Returns:
        A pd.Series indexed by CLASS_NAMES ("SELL", "HOLD", "BUY") giving
        the predicted probability of each, summing to 1.

    Raises:
        ValueError: If `ohlcv` is empty, or if there isn't enough trailing
            history for the most recent row's indicators to be fully
            computed (i.e. any required feature is still NaN).
    """
    if len(ohlcv) == 0:
        raise ValueError("ohlcv must contain at least one row")

    features = build_features(
        ohlcv,
        indicators=indicators,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rsi_length=rsi_length,
        atr_length=atr_length,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
    )

    latest_row = features.iloc[[-1]][FEATURE_COLUMNS]

    if latest_row.isna().any(axis=None):
        raise ValueError(
            "Not enough historical data to compute features for the most "
            "recent row; provide more leading OHLCV history",
        )

    scaled_row = apply_scaler(scaler, latest_row)
    proba = model.predict_proba(scaled_row)

    return proba.iloc[0]