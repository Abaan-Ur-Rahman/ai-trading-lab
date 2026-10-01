"""Classification and trading-oriented evaluation metrics for the ML layer.

Trading-evaluation execution semantics (documented here since they govern
trading_report's behavior):
  - Prediction source: row i's prediction uses only data through candle i
    (guaranteed upstream by the feature layer's own leakage test).
  - Entry: close[i], the same price basis the label was built from.
  - Exit: close[i + horizon] -- the labeling horizon, not a fixed
    next-candle return, since predictions are conditioned on "what happens
    horizon candles out".
  - BUY -> long 1 unit; SELL -> short 1 unit; HOLD -> flat, return 0,
    no cost charged.
  - A trade is only taken if the predicted class's probability >=
    min_confidence; otherwise treated as no-trade regardless of the
    argmax label.
  - transaction_cost_pct is charged once at entry and once at exit
    (2x per completed round-trip), only on rows where a trade was taken.
  - Each row is an independent, fully-collateralized hypothetical trade --
    this is a signal-quality evaluation, not a capital-constrained
    portfolio backtest. Real position sizing belongs to a later
    Backtesting Engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from ml.dataset import CLASS_LABELS, CLASS_NAMES

_LABEL_TO_NAME = dict(zip(CLASS_LABELS, CLASS_NAMES))


def multiclass_brier_score(y_true: pd.Series, y_proba: pd.DataFrame) -> float:
    """Compute the multiclass Brier score.

    BS = (1/N) * sum_i sum_c (p_ic - y_ic)^2

    where y_ic = 1 if sample i's true class is c, else 0 (one-hot), and
    p_ic is the predicted probability of class c for sample i. Lower is
    better; 0 is a perfect probabilistic forecast.
    """
    true_names = pd.Series(y_true).map(_LABEL_TO_NAME)
    one_hot = pd.get_dummies(true_names).reindex(columns=CLASS_NAMES, fill_value=0).astype(float)

    diff = y_proba[CLASS_NAMES].to_numpy() - one_hot.to_numpy()

    return float(np.mean(np.sum(diff**2, axis=1)))


def classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: pd.DataFrame,
) -> dict:
    """Compute classification metrics. Works identically for baselines and
    trained models -- only consumes y_true/y_pred/y_proba, not how they
    were produced.
    """
    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError("y_true, y_pred, and y_proba must be the same length")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for name, p, r, f, s in zip(CLASS_NAMES, precision, recall, f1, support)
    }

    conf_matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    macro_f1 = f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Numeric labels, not string names -- see module docstring / commit
    # notes: sklearn's log_loss silently assumes lexicographic order for
    # string labels regardless of the `labels` argument. CLASS_LABELS is
    # already sorted (-1 < 0 < 1), matching our SELL/HOLD/BUY column order,
    # so this is unambiguous.
    loss = log_loss(y_true, y_proba[CLASS_NAMES].to_numpy(), labels=CLASS_LABELS)

    brier = multiclass_brier_score(y_true, y_proba)

    return {
        "per_class": per_class,
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "log_loss": float(loss),
        "brier_score": brier,
    }


def trading_report(
    y_pred: np.ndarray,
    y_proba: pd.DataFrame,
    ohlcv: pd.DataFrame,
    horizon: int,
    min_confidence: float = 0.5,
    transaction_cost_pct: float = 0.0005,
) -> dict:
    """Simulate independent per-row trades. See module docstring for the
    exact execution semantics this implements.
    """
    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    if not (0 <= min_confidence <= 1):
        raise ValueError("min_confidence must be between 0 and 1")

    if transaction_cost_pct < 0:
        raise ValueError("transaction_cost_pct must be non-negative")

    if not (len(y_pred) == len(y_proba) == len(ohlcv)):
        raise ValueError("y_pred, y_proba, and ohlcv must be the same length")

    close = ohlcv["close"].to_numpy()
    n_rows = len(close)

    name_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    proba_values = y_proba[CLASS_NAMES].to_numpy()

    returns = []
    trades_taken = []

    for i in range(n_rows):
        exit_index = i + horizon
        if exit_index >= n_rows:
            continue  # no realized future return available yet

        direction = int(y_pred[i])
        predicted_name = _LABEL_TO_NAME[direction]
        confidence = proba_values[i, name_to_idx[predicted_name]]

        if direction == 0 or confidence < min_confidence:
            returns.append(0.0)
            trades_taken.append(False)
            continue

        raw_return = (close[exit_index] - close[i]) / close[i]
        signed_return = raw_return if direction == 1 else -raw_return
        net_return = signed_return - 2 * transaction_cost_pct

        returns.append(net_return)
        trades_taken.append(True)

    returns = np.array(returns)
    trades_taken = np.array(trades_taken)

    cumulative_return = float(np.sum(returns))

    std = np.std(returns)
    sharpe_ratio = float(np.mean(returns) / std) if std > 0 else 0.0

    equity_curve = 1.0 + np.cumsum(returns)
    running_max = np.maximum.accumulate(equity_curve) if len(equity_curve) else np.array([])
    drawdowns = (running_max - equity_curve) / running_max if len(equity_curve) else np.array([])
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    n_trades = int(np.sum(trades_taken))
    hit_rate = float(np.mean(returns[trades_taken] > 0)) if n_trades > 0 else 0.0

    return {
        "evaluation_type": "signal-quality evaluation (independent per-row trades), not a portfolio backtest",
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio,
        "sharpe_ratio_note": "raw mean/std ratio over evaluated rows; not annualized (timeframe unknown at this layer)",
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "n_rows_evaluated": int(len(returns)),
        "n_trades_taken": n_trades,
    }