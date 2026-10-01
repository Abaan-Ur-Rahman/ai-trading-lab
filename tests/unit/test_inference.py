"""Unit tests for predict_from_ohlcv."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.scaling import apply_scaler, fit_scaler
from ml.dataset import CLASS_NAMES, FEATURE_COLUMNS
from ml.inference import predict_from_ohlcv
from ml.models.logistic_regression import LogisticRegressionModel


@pytest.fixture
def ohlcv_with_enough_history() -> pd.DataFrame:
    """Enough rows that every default indicator window is fully warmed up."""
    n = 80
    rng = np.random.default_rng(3)
    closes = 100 + rng.normal(size=n).cumsum()
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": rng.integers(100, 1000, size=n),
        }
    )


@pytest.fixture
def fitted_model_and_scaler():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.Series([-1] * 20 + [0] * 20 + [1] * 20, name="label")

    scaler = fit_scaler(X)
    X_scaled = apply_scaler(scaler, X)

    model = LogisticRegressionModel(random_state=1)
    model.fit(X_scaled, y)

    return model, scaler


def test_predict_from_ohlcv_returns_series_summing_to_one(
    ohlcv_with_enough_history, fitted_model_and_scaler
) -> None:
    model, scaler = fitted_model_and_scaler

    result = predict_from_ohlcv(ohlcv_with_enough_history, model, scaler)

    assert isinstance(result, pd.Series)
    assert list(result.index) == CLASS_NAMES
    assert result.sum() == pytest.approx(1.0)


def test_predict_from_ohlcv_rejects_empty_ohlcv(fitted_model_and_scaler) -> None:
    model, scaler = fitted_model_and_scaler
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    with pytest.raises(ValueError):
        predict_from_ohlcv(empty, model, scaler)


def test_predict_from_ohlcv_rejects_insufficient_history(fitted_model_and_scaler) -> None:
    """Too few rows for indicators to warm up should raise, not predict on NaN."""
    model, scaler = fitted_model_and_scaler
    short_history = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
            "volume": [500, 500, 500],
        }
    )

    with pytest.raises(ValueError):
        predict_from_ohlcv(short_history, model, scaler)


def test_predict_from_ohlcv_does_not_mutate_input(
    ohlcv_with_enough_history, fitted_model_and_scaler
) -> None:
    model, scaler = fitted_model_and_scaler
    original = ohlcv_with_enough_history.copy()

    predict_from_ohlcv(ohlcv_with_enough_history, model, scaler)

    pd.testing.assert_frame_equal(ohlcv_with_enough_history, original)