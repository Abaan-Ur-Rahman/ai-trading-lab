"""Forward test of the production model on bars that arrive after it was trained.

Two commands:

    python scripts/forward_test.py update   # run daily: fetch new bars, log predictions
    python scripts/forward_test.py report   # score everything logged so far

`update` fetches the latest complete 1h bars for gold, EUR/USD and USD/JPY,
appends them to data/forward/ (data/raw/ stays frozen, because every recorded
result splits that data by position and would shift if rows were added),
then predicts every new gold bar with the saved production model exactly as
live use would -- predict_from_ohlcv on history up to that bar only -- and
appends the probabilities to reports/forward/predictions.csv. That log is
append-only: an existing prediction is never rewritten, and each row records
when it was logged.

`report` attaches what actually happened 5 bars later and evaluates:
classification quality vs an always-HOLD baseline, and a capital-constrained
backtest of each pre-registered rule below vs buy-and-hold.

PRE-REGISTERED RULES (fixed 2026-10-06, before any forward bar was evaluated).
Do not add, remove or edit them after looking at forward results; a new idea
gets its own registration date and is judged only on bars after that date.
- A_current: the backtest's rules unchanged (5-bar hold, confidence >= 0.50).
- B_confident: confidence >= 0.55. 0.55 rather than 0.60 because only 14% of
  backtest trades reached 0.60 (too few to judge within months); chosen for
  trade frequency, without looking at returns by confidence.
- C_long_only: no shorts (gold rose over the history; longs had the larger
  gross edge in the backtest's seed-0 trades).
- D_longer_hold: hold 24 bars, so each move is larger relative to costs.
Verdict rule: a candidate earns further work only once at least
MIN_BARS_FOR_VERDICT forward bars have outcomes, and only if its total return
after costs AND its Sharpe ratio are both positive. Four candidates are being
compared, so one of them clearing the bar by a small margin may be luck.

AMENDMENT (2026-10-06, the day the rules were registered, with 104 forward bars
logged and before any verdict-relevant amount of data): Twelve Data serves 1h
quotes 24/7, weekends included, and nobody can trade gold or these currency
pairs at weekend prices. In the backtest, weekend-touching trades were roughly
break-even while weekday trades lost more, so weekend quotes can flatter a
result. A rule therefore now ALSO has to pass the same test using only trades
whose signal, entry and exit bars are all Monday-Friday UTC
(forward.tracking.weekday_tradeable_signals). This only makes the verdict
stricter, never looser. Because bars arrive 24/7, 1,000 bars take about six
weeks (around mid-November 2026), not two months.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from backtest.engine import BacktestConfig, buy_and_hold, run_backtest
from data.dataframe import candles_to_dataframe
from forward.tracking import (
    append_bars,
    attach_outcomes,
    bars_ready_to_predict,
    complete_new_bars,
    predicted_class,
    weekday_tradeable_signals,
)
from ml.dataset import CLASS_NAMES
from ml.inference import predict_from_ohlcv
from ml.persistence import load_model

MODEL_DIRECTORY = PROJECT_ROOT / "models/XAUUSD_1h_rf_balanced_plus_combined_production_v1"
PRIMARY = ("XAUUSD", "XAU/USD")
RAW_DIR = PROJECT_ROOT / "data/raw"
FORWARD_DIR = PROJECT_ROOT / "data/forward"
REPORT_DIR = PROJECT_ROOT / "reports/forward"
PREDICTIONS_PATH = REPORT_DIR / "predictions.csv"
TIMEFRAME = "1h"
MAX_FETCH = 5000  # Twelve Data's per-request maximum
SECONDS_BETWEEN_REQUESTS = 8  # stays under the free plan's 8 requests/minute
MIN_BARS_FOR_VERDICT = 1000  # about 6 weeks: Twelve Data serves 1h bars 24/7
HORIZON = 5
THRESHOLD = 0.005

RULES_REGISTERED_ON = "2026-10-06"
RULES = {
    "A_current": BacktestConfig(min_confidence=0.50, holding_bars=5),
    "B_confident": BacktestConfig(min_confidence=0.55, holding_bars=5),
    "C_long_only": BacktestConfig(min_confidence=0.50, holding_bars=5, allow_short=False),
    "D_longer_hold": BacktestConfig(min_confidence=0.50, holding_bars=24),
}


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp").sort_index()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="timestamp")


def instruments(metadata) -> dict[str, str]:
    """{name: provider symbol} for the primary and every secondary the model needs."""
    return {PRIMARY[0]: PRIMARY[1], **metadata.secondary_symbols}


def history(name: str) -> pd.DataFrame:
    """Frozen raw history plus any forward bars, as one chronological frame."""
    raw = load_csv(RAW_DIR / f"{name}_{TIMEFRAME}.csv")
    if raw is None:
        sys.exit(f"Missing {RAW_DIR / f'{name}_{TIMEFRAME}.csv'}; the forward test extends the raw history.")
    forward = load_csv(FORWARD_DIR / f"{name}_{TIMEFRAME}.csv")
    return raw if forward is None else pd.concat([raw, forward.loc[forward.index > raw.index.max()]])


def fetch_new_bars(provider, names: dict[str, str], now: pd.Timestamp) -> None:
    for i, (name, symbol) in enumerate(names.items()):
        if i:
            time.sleep(SECONDS_BETWEEN_REQUESTS)
        stored = history(name)
        last = stored.index.max()
        needed = int((now - last) / pd.Timedelta(hours=1)) + 10
        if needed > MAX_FETCH:
            print(f"  {name}: {needed} bars missing, more than one request can fetch; fetching the latest "
                  f"{MAX_FETCH}. Run update more often to avoid gaps.")
        candles = provider.get_candles(symbol=symbol, timeframe=TIMEFRAME, limit=min(max(needed, 50), MAX_FETCH))
        new = complete_new_bars(candles_to_dataframe(candles), after=last, now=now)
        forward_path = FORWARD_DIR / f"{name}_{TIMEFRAME}.csv"
        updated = append_bars(load_csv(forward_path), new)
        if not updated.empty:
            save_csv(updated, forward_path)
        print(f"  {name}: {len(new)} new complete bar(s) stored (latest {updated.index.max() if not updated.empty else last})")


def update(provider=None) -> None:
    model, scaler, metadata = load_model(MODEL_DIRECTORY)
    names = instruments(metadata)
    now = pd.Timestamp.now(tz="UTC")

    if provider is None:
        from data.providers.twelve_data import TwelveDataProvider

        provider = TwelveDataProvider()
    print(f"Fetching new bars ({now:%Y-%m-%d %H:%M} UTC)...")
    fetch_new_bars(provider, names, now)

    frames = {name: history(name) for name in names}
    primary = frames[PRIMARY[0]]
    secondaries = {name: frames[name] for name in metadata.secondary_symbols}
    start_after = load_csv(RAW_DIR / f"{PRIMARY[0]}_{TIMEFRAME}.csv").index.max()

    log = load_csv(PREDICTIONS_PATH)
    logged = log.index if log is not None else pd.DatetimeIndex([], tz="UTC")
    to_predict = bars_ready_to_predict(
        primary.index, {name: frame.index.max() for name, frame in secondaries.items()}, logged, start_after,
    )

    rows, skipped = [], []
    for bar in to_predict:
        # One bar whose inputs can't be used (e.g. a currency feed with a gap of
        # more than 24h, which predict_from_ohlcv refuses) must not stop every
        # other bar from being logged, today and on every later run. It is
        # skipped, reported here, and retried on the next run; since stored
        # history never changes, a bar that can't be predicted stays unlogged,
        # exactly as a live trader would have had no valid signal for it.
        try:
            proba = predict_from_ohlcv(
                primary.loc[:bar], model, scaler,
                feature_columns=metadata.feature_columns,
                secondary_ohlcv={name: frame.loc[:bar] for name, frame in secondaries.items()},
                cross_asset_corr_window=metadata.cross_asset_corr_window,
            )
        except ValueError as exc:
            skipped.append(bar)
            print(f"  WARNING: skipped {bar:%Y-%m-%d %H:%M}: {exc}")
            continue
        rows.append({"timestamp": bar, **proba.to_dict(), "close": primary.loc[bar, "close"],
                     "model": MODEL_DIRECTORY.name, "logged_at": now.isoformat()})

    if rows:
        new_rows = pd.DataFrame(rows).set_index("timestamp")
        save_csv(new_rows if log is None else pd.concat([log, new_rows]), PREDICTIONS_PATH)
        latest = new_rows.iloc[-1]
        print(f"Logged {len(rows)} new prediction(s). Latest, {new_rows.index[-1]:%Y-%m-%d %H:%M}: "
              + ", ".join(f"{c} {latest[c]:.3f}" for c in CLASS_NAMES))
    else:
        print("No new bars to predict yet.")
    if skipped:
        print(f"WARNING: {len(skipped)} bar(s) could not be predicted (see above); they will be retried next run.")
    total = 0 if log is None else len(log)
    print(f"Prediction log: {total + len(rows)} bar(s) since {start_after:%Y-%m-%d %H:%M}.")


def evaluate_rule(prices: pd.DataFrame, signals: pd.DataFrame, config: BacktestConfig) -> dict:
    """Backtest one rule on all forward bars and on weekday-only trades, plus its verdict inputs."""
    all_bars = run_backtest(prices, signals, config).metrics
    weekday_signals = weekday_tradeable_signals(signals, prices.index, config.holding_bars)
    weekday = run_backtest(prices, weekday_signals, config).metrics if not weekday_signals.empty else None
    passes_all = all_bars["total_return"] > 0 and all_bars["sharpe_ratio"] > 0
    passes_weekday = weekday is not None and weekday["total_return"] > 0 and weekday["sharpe_ratio"] > 0
    return {"all_bars": all_bars, "weekday_only": weekday, "passes": passes_all and passes_weekday}


def report() -> None:
    log = load_csv(PREDICTIONS_PATH)
    if log is None or log.empty:
        sys.exit("No predictions logged yet. Run: python scripts/forward_test.py update")

    primary = history(PRIMARY[0])
    scored = attach_outcomes(log, primary, horizon=HORIZON, threshold=THRESHOLD)
    done = scored.dropna(subset=["actual"])
    print(f"Forward bars logged: {len(scored)}; with known outcomes: {len(done)} "
          f"({len(done) / MIN_BARS_FOR_VERDICT:.0%} of the {MIN_BARS_FOR_VERDICT} needed for a verdict).")
    if done.empty:
        print("Outcomes appear 5 bars after each prediction; run update again later.")
        return

    predicted = predicted_class(done)
    actual = done["actual"].astype(int)
    model_f1 = f1_score(actual, predicted, labels=[-1, 0, 1], average="macro", zero_division=0)
    hold_f1 = f1_score(actual, np.zeros(len(actual), dtype=int), labels=[-1, 0, 1], average="macro", zero_division=0)
    print(f"\nClassification (bars with outcomes): macro-F1 {model_f1:.4f} vs always-HOLD {hold_f1:.4f}; "
          f"accuracy {np.mean(predicted.to_numpy() == actual.to_numpy()):.1%}")

    prices = primary.loc[done.index.min():]
    signals = done[CLASS_NAMES]
    benchmark = buy_and_hold(prices, done.index.min())
    verdict_ready = len(done) >= MIN_BARS_FOR_VERDICT

    print(f"\nPre-registered rules (registered {RULES_REGISTERED_ON}), backtested on forward bars:")
    print(f"{'rule':<16}{'return':>9}{'sharpe':>8}{'max DD':>8}{'trades':>8}{'win rate':>10}"
          f"{'weekday ret':>13}{'wkday shrp':>11}  verdict")
    summary = {}
    for name, config in RULES.items():
        evaluation = evaluate_rule(prices, signals, config)
        m, w = evaluation["all_bars"], evaluation["weekday_only"]
        verdict = ("PASSES" if evaluation["passes"] else "fails") if verdict_ready else "too early"
        weekday_cells = f"{w['total_return']:>+13.1%}{w['sharpe_ratio']:>+11.2f}" if w else f"{'no trades':>13}{'':>11}"
        print(f"{name:<16}{m['total_return']:>+9.1%}{m['sharpe_ratio']:>+8.2f}{m['max_drawdown']:>8.1%}"
              f"{m['n_trades']:>8}{m['win_rate']:>10.1%}{weekday_cells}  {verdict}")
        summary[name] = {"rules": asdict(config), "metrics": m, "weekday_only_metrics": w, "verdict": verdict}
    bm = benchmark.metrics
    print(f"{'buy_and_hold':<16}{bm['total_return']:>+9.1%}{bm['sharpe_ratio']:>+8.2f}{bm['max_drawdown']:>8.1%}")
    print("A rule passes only if return and Sharpe are positive on all bars AND on weekday-only trades.")
    if not verdict_ready:
        print(f"\nToo early to judge: wait for {MIN_BARS_FOR_VERDICT} bars with outcomes. "
              "Early numbers swing a lot; don't act on them.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.json").write_text(json.dumps({
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "bars_logged": len(scored),
        "bars_with_outcomes": len(done),
        "period": [str(done.index.min()), str(done.index.max())],
        "macro_f1": model_f1,
        "always_hold_macro_f1": hold_f1,
        "rules_registered_on": RULES_REGISTERED_ON,
        "rules": summary,
        "buy_and_hold": bm,
        "verdict_ready": verdict_ready,
    }, indent=2, default=str))
    print(f"\nSaved {REPORT_DIR / 'report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["update", "report"])
    args = parser.parse_args()
    update() if args.command == "update" else report()


if __name__ == "__main__":
    main()
