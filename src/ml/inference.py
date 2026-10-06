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

from features.cross_asset import build_prefixed_cross_asset_features
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
    range_lookback: int = 20,
    feature_columns: list[str] | None = None,
    secondary_ohlcv: dict[str, pd.DataFrame] | None = None,
    cross_asset_corr_window: int = 20,
    max_secondary_staleness: pd.Timedelta = pd.Timedelta(hours=24),
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
            macd_signal, range_lookback: Must match the values used when
            the model's training dataset was built, or the computed
            features will not mean what the model was trained on.
        feature_columns: The columns the model was trained on, in order
            (a saved model's metadata.feature_columns). Defaults to
            FEATURE_COLUMNS, the standard 5-feature set.
        secondary_ohlcv: For a model trained with cross-asset features:
            {name: OHLCV} for each secondary instrument, keyed by the same
            names as the model's metadata.secondary_symbols (e.g. "EURUSD").
            Each must cover the same period as `ohlcv`; all frames must be
            indexed by timestamp so they can be aligned.
        cross_asset_corr_window: Must match the model's
            metadata.cross_asset_corr_window.
        max_secondary_staleness: Refuse to predict if a secondary's latest
            bar is older than the primary's latest bar by more than this.
            Secondary prices are forward-filled onto the primary's
            timestamps, so a feed that stopped updating would otherwise
            silently feed stale values into the prediction.

    Returns:
        A pd.Series indexed by CLASS_NAMES ("SELL", "HOLD", "BUY") giving
        the predicted probability of each, summing to 1.

    Raises:
        ValueError: If `ohlcv` is empty, if secondary data is given without
            timestamp indexes or is stale (see max_secondary_staleness), or if there isn't enough trailing history for
            the most recent row's features to be fully computed (i.e. any
            required feature is still NaN).
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
        range_lookback=range_lookback,
    )

    if secondary_ohlcv:
        frames = [ohlcv, *secondary_ohlcv.values()]
        if not all(isinstance(frame.index, pd.DatetimeIndex) for frame in frames):
            raise ValueError("ohlcv and every secondary_ohlcv frame must have a DatetimeIndex")
        latest = ohlcv.index.max()
        for name, frame in secondary_ohlcv.items():
            if latest - frame.index.max() > max_secondary_staleness:
                raise ValueError(
                    f"Secondary {name} data ends at {frame.index.max()}, more than "
                    f"{max_secondary_staleness} before the primary's latest bar ({latest}); "
                    "refresh it before predicting",
                )
        cross_asset = build_prefixed_cross_asset_features(
            ohlcv["close"],
            {name: frame["close"] for name, frame in secondary_ohlcv.items()},
            corr_window=cross_asset_corr_window,
        )
        features = pd.concat([features, cross_asset], axis=1)

    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns
    missing = [column for column in columns if column not in features.columns]
    if missing:
        raise ValueError(
            f"Cannot build feature columns {missing}; a cross-asset model needs "
            "secondary_ohlcv for every secondary instrument it was trained with",
        )

    latest_row = features.iloc[[-1]][columns]

    if latest_row.isna().any(axis=None):
        raise ValueError(
            "Not enough historical data to compute features for the most "
            "recent row; provide more leading OHLCV history",
        )

    scaled_row = apply_scaler(scaler, latest_row)
    proba = model.predict_proba(scaled_row)

    return proba.iloc[0]