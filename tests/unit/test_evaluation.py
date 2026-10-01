"""Unit tests for classification and trading evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.evaluation import classification_report, multiclass_brier_score, trading_report


@pytest.fixture
def sample_predictions() -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """Hand-verified fixture: predictions cross-checked against sklearn directly."""
    y_true = pd.Series([-1, 0, 1, 1, -1])
    y_pred = np.array([-1, 0, 1, -1, -1])
    y_proba = pd.DataFrame(
        {
            "SELL": [0.7, 0.1, 0.2, 0.6, 0.5],
            "HOLD": [0.2, 0.8, 0.1, 0.1, 0.3],
            "BUY": [0.1, 0.1, 0.7, 0.3, 0.2],
        }
    )
    return y_true, y_pred, y_proba


def test_classification_report_matches_sklearn_directly(
    sample_predictions: tuple[pd.Series, np.ndarray, pd.DataFrame],
) -> None:
    """Every metric should match values computed directly against sklearn."""
    y_true, y_pred, y_proba = sample_predictions

    report = classification_report(y_true, y_pred, y_proba)

    assert report["per_class"]["SELL"]["precision"] == pytest.approx(0.6667, abs=1e-3)
    assert report["per_class"]["SELL"]["recall"] == pytest.approx(1.0)
    assert report["per_class"]["HOLD"]["f1"] == pytest.approx(1.0)
    assert report["per_class"]["BUY"]["recall"] == pytest.approx(0.5)

    assert report["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert report["macro_f1"] == pytest.approx(0.8222, abs=1e-3)
    assert report["accuracy"] == pytest.approx(0.8)
    assert report["log_loss"] == pytest.approx(0.5667, abs=1e-3)
    assert report["brier_score"] == pytest.approx(0.316, abs=1e-3)


def test_classification_report_requires_matching_lengths() -> None:
    """y_true, y_pred, and y_proba must all be the same length."""
    y_true = pd.Series([-1, 0])
    y_pred = np.array([-1])
    y_proba = pd.DataFrame({"SELL": [0.5], "HOLD": [0.3], "BUY": [0.2]})

    with pytest.raises(ValueError):
        classification_report(y_true, y_pred, y_proba)


def test_multiclass_brier_score_matches_hand_formula() -> None:
    """A perfect forecast should score exactly 0."""
    y_true = pd.Series([-1, 0, 1])
    y_proba = pd.DataFrame(
        {"SELL": [1.0, 0.0, 0.0], "HOLD": [0.0, 1.0, 0.0], "BUY": [0.0, 0.0, 1.0]}
    )

    score = multiclass_brier_score(y_true, y_proba)

    assert score == pytest.approx(0.0)


def test_trading_report_matches_hand_computed_values() -> None:
    """Cumulative return, Sharpe, drawdown, and hit rate verified against hand-fabricated prices."""
    ohlcv = pd.DataFrame({"close": [100.0, 102.0, 101.0, 105.0, 103.0, 108.0]})
    y_pred = np.array([1, -1, 0, 1, 0, 0])
    y_proba = pd.DataFrame(
        {
            "SELL": [0.05, 0.4, 0.3, 0.1, 0.3, 0.3],
            "HOLD": [0.05, 0.3, 0.5, 0.3, 0.4, 0.4],
            "BUY": [0.9, 0.3, 0.2, 0.6, 0.3, 0.3],
        }
    )

    report = trading_report(
        y_pred, y_proba, ohlcv, horizon=2, min_confidence=0.5, transaction_cost_pct=0.001,
    )

    assert report["cumulative_return"] == pytest.approx(0.034571, abs=1e-5)
    assert report["sharpe_ratio"] == pytest.approx(0.796277, abs=1e-5)
    assert report["max_drawdown"] == pytest.approx(0.0)
    assert report["hit_rate"] == pytest.approx(1.0)
    assert report["n_trades_taken"] == 2
    assert report["n_rows_evaluated"] == 4


def test_trading_report_rejects_non_positive_horizon() -> None:
    """horizon must be greater than 0."""
    ohlcv = pd.DataFrame({"close": [100.0, 101.0]})
    y_pred = np.array([0, 0])
    y_proba = pd.DataFrame({"SELL": [0.3, 0.3], "HOLD": [0.4, 0.4], "BUY": [0.3, 0.3]})

    with pytest.raises(ValueError):
        trading_report(y_pred, y_proba, ohlcv, horizon=0)


def test_trading_report_rejects_invalid_min_confidence() -> None:
    """min_confidence must be between 0 and 1."""
    ohlcv = pd.DataFrame({"close": [100.0, 101.0]})
    y_pred = np.array([0, 0])
    y_proba = pd.DataFrame({"SELL": [0.3, 0.3], "HOLD": [0.4, 0.4], "BUY": [0.3, 0.3]})

    with pytest.raises(ValueError):
        trading_report(y_pred, y_proba, ohlcv, horizon=1, min_confidence=1.5)