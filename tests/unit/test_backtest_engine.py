"""Tests for backtest.engine, using small hand-checkable price paths."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import BacktestConfig, buy_and_hold, run_backtest, signal_direction


def _prices(opens: list[float], closes: list[float] | None = None) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(opens), freq="h", tz="UTC")
    return pd.DataFrame({"open": opens, "close": closes if closes is not None else opens}, index=index)


def _signal(prices: pd.DataFrame, positions: list[int], sell: float, hold: float, buy: float) -> pd.DataFrame:
    rows = [{"SELL": sell, "HOLD": hold, "BUY": buy}] * len(positions)
    return pd.DataFrame(rows, index=prices.index[positions])


NO_COST = BacktestConfig(holding_bars=3, cost_pct_per_side=0.0)


def test_long_enters_next_open_and_exits_after_holding_bars() -> None:
    prices = _prices([100, 100, 102, 104, 110, 110, 110], closes=[100, 101, 103, 105, 110, 110, 110])

    result = run_backtest(prices, _signal(prices, [0], 0.1, 0.2, 0.7), NO_COST)

    trade = result.trades.iloc[0]
    assert trade["entry_time"] == prices.index[1]  # next bar's open, not the signal bar
    assert trade["exit_time"] == prices.index[4]  # 3 bars later, at the open
    assert trade["entry_price"] == 100 and trade["exit_price"] == 110
    assert trade["pnl"] == pytest.approx(1_000)  # 100 units x +10
    # Marked to market at each close while open: 101, 103, 105.
    assert result.equity.iloc[1:4].tolist() == pytest.approx([10_100, 10_300, 10_500])
    assert result.equity.iloc[-1] == pytest.approx(11_000)


def test_short_profits_when_price_falls() -> None:
    prices = _prices([100, 100, 95, 90, 90, 90])

    result = run_backtest(prices, _signal(prices, [0], 0.8, 0.1, 0.1), NO_COST)

    assert result.trades.iloc[0]["direction"] == -1
    assert result.trades.iloc[0]["pnl"] == pytest.approx(1_000)  # 100 units x -10, short


def test_costs_charged_on_entry_and_exit_notional() -> None:
    prices = _prices([100, 100, 100, 100, 110, 110])
    config = BacktestConfig(holding_bars=3, cost_pct_per_side=0.001)

    trade = run_backtest(prices, _signal(prices, [0], 0.1, 0.2, 0.7), config).trades.iloc[0]

    # Entry: 10,000 x 0.1% = 10. Exit: 100 units x 110 x 0.1% = 11.
    assert trade["costs"] == pytest.approx(21)
    assert trade["pnl"] == pytest.approx(1_000 - 21)


def test_signals_ignored_while_a_position_is_open() -> None:
    prices = _prices([100] * 10)

    result = run_backtest(prices, _signal(prices, [0, 1, 2], 0.1, 0.2, 0.7), NO_COST)

    assert len(result.trades) == 1


def test_gains_compound_into_next_position_size() -> None:
    prices = _prices([100, 100, 100, 100, 200, 200, 200, 200, 200, 200])

    result = run_backtest(prices, _signal(prices, [0, 4], 0.1, 0.2, 0.7), NO_COST)

    assert result.trades.iloc[1]["notional"] == pytest.approx(20_000)


def test_low_confidence_and_hold_signals_do_not_trade() -> None:
    prices = _prices([100] * 8)
    signals = pd.concat([
        _signal(prices, [0], 0.1, 0.45, 0.45),  # BUY tied with HOLD but below 0.5
        _signal(prices, [2], 0.1, 0.8, 0.1),  # HOLD
    ])

    result = run_backtest(prices, signals, BacktestConfig(holding_bars=2))

    assert result.trades.empty
    assert result.metrics["n_trades"] == 0


def test_shorts_skipped_when_not_allowed() -> None:
    assert signal_direction(pd.Series({"SELL": 0.8, "HOLD": 0.1, "BUY": 0.1}), 0.5, allow_short=False) == (0, 0.8)
    assert signal_direction(pd.Series({"SELL": 0.8, "HOLD": 0.1, "BUY": 0.1}), 0.5, allow_short=True) == (-1, 0.8)


def test_no_entry_when_holding_period_would_run_past_the_data() -> None:
    prices = _prices([100] * 4)

    result = run_backtest(prices, _signal(prices, [1], 0.1, 0.2, 0.7), NO_COST)

    assert result.trades.empty


def test_max_drawdown_from_marked_equity() -> None:
    prices = _prices([100, 100, 100, 80, 120, 120], closes=[100, 100, 100, 80, 120, 120])

    result = run_backtest(prices, _signal(prices, [0], 0.1, 0.2, 0.7), NO_COST)

    assert result.metrics["max_drawdown"] == pytest.approx(0.2)


def test_rejects_signals_outside_prices() -> None:
    prices = _prices([100] * 5)
    stray = pd.DataFrame({"SELL": [0.1], "HOLD": [0.2], "BUY": [0.7]},
                         index=[pd.Timestamp("2030-01-01", tz="UTC")])

    with pytest.raises(ValueError):
        run_backtest(prices, stray)


def test_config_rejects_leverage() -> None:
    with pytest.raises(ValueError):
        BacktestConfig(position_fraction=2.0)


def test_buy_and_hold_tracks_price_from_next_open() -> None:
    prices = _prices([100, 100, 110, 120], closes=[100, 105, 115, 125])

    result = buy_and_hold(prices, prices.index[0], BacktestConfig(cost_pct_per_side=0.0))

    assert result.equity.tolist() == pytest.approx([10_000, 10_500, 11_500, 12_500])
