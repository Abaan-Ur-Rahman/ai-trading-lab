"""Capital-constrained, bar-by-bar backtest of SELL/HOLD/BUY probability signals.

Unlike ml.evaluation.trading_report (independent per-row trades, returns
simply summed), this simulates one account over time:

- One position at a time. While a position is open, new signals are ignored,
  so overlapping trades can't be double-counted.
- No look-ahead in execution. A signal is formed from bar t's features, which
  use bar t's close, so the earliest realistic fill is the OPEN of bar t+1.
  The position is closed at the open of the bar `holding_bars` later --
  entering at open[t+1] and exiting at open[t+1+holding_bars] approximates the
  close[t] -> close[t+holding_bars] move the labels were defined on.
- Position size is a fixed fraction of current equity at entry (1.0 = fully
  invested, no leverage), so gains and losses compound.
- Costs are charged on notional at entry and at exit (spread + commission).

Equity is marked to market at every bar's close. Metrics are computed from
that equity curve and from the closed-trade list.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ml.dataset import CLASS_NAMES

_DIRECTION_BY_CLASS = {"SELL": -1, "HOLD": 0, "BUY": 1}


@dataclass(frozen=True)
class BacktestConfig:
    """Trading rules. Fix these before looking at results; tuning them on the
    same period you report would turn the backtest into an optimisation."""

    initial_capital: float = 10_000.0
    position_fraction: float = 1.0
    min_confidence: float = 0.5
    holding_bars: int = 5
    cost_pct_per_side: float = 0.0005
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]; leverage is not modelled")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.holding_bars <= 0:
            raise ValueError("holding_bars must be positive")
        if self.cost_pct_per_side < 0:
            raise ValueError("cost_pct_per_side must be non-negative")


@dataclass(frozen=True)
class Trade:
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    confidence: float
    entry_price: float
    exit_price: float
    notional: float
    costs: float
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def signal_direction(proba_row: pd.Series, min_confidence: float, allow_short: bool) -> tuple[int, float]:
    """Direction (-1, 0, 1) and confidence for one row of class probabilities.

    Takes the most likely class; trades only if it is BUY or SELL with
    probability at least `min_confidence` (and SELL only if shorting is allowed).
    """
    predicted = proba_row[CLASS_NAMES].astype(float).idxmax()
    confidence = float(proba_row[predicted])
    direction = _DIRECTION_BY_CLASS[predicted]
    if direction == 0 or confidence < min_confidence or (direction == -1 and not allow_short):
        return 0, confidence
    return direction, confidence


def run_backtest(prices: pd.DataFrame, signals: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    """Simulate trading `signals` on `prices`.

    Args:
        prices: OHLC bars with at least `open` and `close`, indexed by
            timestamp, chronologically ordered. The simulation covers every
            bar from the first signal's bar to the last bar.
        signals: Class probabilities with columns SELL/HOLD/BUY, indexed by
            the bar whose close they were formed at. Must be a subset of
            `prices.index`. Bars without a signal never open a position.
        config: Trading rules; defaults to BacktestConfig().
    """
    config = config or BacktestConfig()

    missing_columns = {"open", "close"}.difference(prices.columns)
    if missing_columns:
        raise ValueError(f"prices is missing columns: {sorted(missing_columns)}")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be chronologically ordered")
    if signals.empty:
        raise ValueError("signals must not be empty")
    if not signals.index.isin(prices.index).all():
        raise ValueError("every signal timestamp must be a bar in prices")

    window = prices.loc[signals.index.min():]
    opens = window["open"].to_numpy(dtype=float)
    closes = window["close"].to_numpy(dtype=float)
    times = window.index
    signal_lookup = signals.reindex(times)
    has_signal = signal_lookup[CLASS_NAMES].notna().all(axis=1).to_numpy()

    cash = config.initial_capital  # equity excluding any open position's unrealised P&L
    equity = np.empty(len(window))
    trades: list[Trade] = []
    position = None  # dict while a position is open

    for i in range(len(window)):
        # 1) Exit at this bar's open if the holding period is up.
        if position is not None and i == position["exit_index"]:
            exit_price = opens[i]
            gross = position["direction"] * position["units"] * (exit_price - position["entry_price"])
            exit_cost = position["units"] * exit_price * config.cost_pct_per_side
            pnl = gross - position["entry_cost"] - exit_cost
            cash += gross - exit_cost  # entry cost was already taken from cash at entry
            trades.append(Trade(
                signal_time=times[position["signal_index"]],
                entry_time=times[position["entry_index"]],
                exit_time=times[i],
                direction=position["direction"],
                confidence=position["confidence"],
                entry_price=position["entry_price"],
                exit_price=exit_price,
                notional=position["notional"],
                costs=position["entry_cost"] + exit_cost,
                pnl=pnl,
                return_pct=pnl / position["equity_at_entry"],
            ))
            position = None

        # 2) Enter at this bar's open on the previous bar's signal.
        if position is None and i > 0 and has_signal[i - 1]:
            direction, confidence = signal_direction(
                signal_lookup.iloc[i - 1], config.min_confidence, config.allow_short,
            )
            exit_index = i + config.holding_bars
            if direction != 0 and exit_index < len(window):
                notional = cash * config.position_fraction
                entry_cost = notional * config.cost_pct_per_side
                cash -= entry_cost
                position = {
                    "signal_index": i - 1,
                    "entry_index": i,
                    "exit_index": exit_index,
                    "direction": direction,
                    "confidence": confidence,
                    "entry_price": opens[i],
                    "units": notional / opens[i],
                    "notional": notional,
                    "entry_cost": entry_cost,
                    "equity_at_entry": cash + entry_cost,
                }

        # 3) Mark to market at this bar's close.
        unrealised = 0.0
        if position is not None:
            unrealised = position["direction"] * position["units"] * (closes[i] - position["entry_price"])
        equity[i] = cash + unrealised

    equity_series = pd.Series(equity, index=times, name="equity")
    trades_frame = pd.DataFrame([asdict(trade) for trade in trades])
    return BacktestResult(
        equity=equity_series,
        trades=trades_frame,
        metrics=compute_metrics(equity_series, trades_frame, config),
    )


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, config: BacktestConfig) -> dict:
    """Summary statistics from an equity curve and its closed trades."""
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    bar_returns = equity.pct_change().dropna()
    bars_per_year = len(equity) / years if years > 0 else float("nan")

    running_peak = equity.cummax()
    drawdown = (running_peak - equity) / running_peak
    final = float(equity.iloc[-1])

    metrics = {
        "initial_capital": config.initial_capital,
        "final_equity": final,
        "total_return": final / config.initial_capital - 1,
        "annualized_return": (final / config.initial_capital) ** (1 / years) - 1 if years > 0 else float("nan"),
        "annualized_volatility": float(bar_returns.std() * np.sqrt(bars_per_year)),
        "sharpe_ratio": (
            float(bar_returns.mean() / bar_returns.std() * np.sqrt(bars_per_year))
            if bar_returns.std() > 0 else 0.0
        ),
        "max_drawdown": float(drawdown.max()),
        "n_trades": int(len(trades)),
        "years": years,
    }

    if len(trades):
        wins, losses = trades.loc[trades["pnl"] > 0, "pnl"], trades.loc[trades["pnl"] <= 0, "pnl"]
        metrics.update({
            "n_long": int((trades["direction"] == 1).sum()),
            "n_short": int((trades["direction"] == -1).sum()),
            "win_rate": float((trades["pnl"] > 0).mean()),
            "avg_trade_return": float(trades["return_pct"].mean()),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
            "total_costs": float(trades["costs"].sum()),
            "exposure": float(
                sum(((equity.index >= t.entry_time) & (equity.index < t.exit_time)).sum() for t in trades.itertuples())
                / len(equity)
            ),
        })
    else:
        metrics.update({"n_long": 0, "n_short": 0, "win_rate": 0.0, "avg_trade_return": 0.0,
                        "profit_factor": float("nan"), "total_costs": 0.0, "exposure": 0.0})
    return metrics


def buy_and_hold(prices: pd.DataFrame, start: pd.Timestamp, config: BacktestConfig | None = None) -> BacktestResult:
    """Benchmark: fully invested long from the open of the bar after `start`, held to the last close."""
    config = config or BacktestConfig()
    window = prices.loc[start:]
    entry_price = float(window["open"].iloc[1])
    notional = config.initial_capital
    entry_cost = notional * config.cost_pct_per_side
    units = notional / entry_price
    marks = config.initial_capital - entry_cost + units * (window["close"].iloc[1:] - entry_price)
    equity = pd.concat([pd.Series([config.initial_capital], index=window.index[:1]), marks]).rename("equity")
    return BacktestResult(equity=equity, trades=pd.DataFrame(), metrics=compute_metrics(equity, pd.DataFrame(), config))
