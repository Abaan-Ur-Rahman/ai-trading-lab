"""Walk-forward, capital-constrained backtest: base vs plus_combined vs buy-and-hold.

Why walk-forward: the production model was trained on all data, so trading
its predictions on that same data would be in-sample and meaningless. Here
every prediction comes from a model that only saw the past:

- From the first backtest bar (the walk-forward sweep's first validation bar,
  2025-04-28 by default) onward, the model is retrained every
  RETRAIN_EVERY_BARS bars (~1 month of 1h bars) on all rows before that
  point, minus a HORIZON-row purge so no training label peeks into the
  prediction block, then predicts the next block.
- Same fixed RF hyperparameters as every earlier check, no tuning.
- Each configuration is run over several forest seeds; metrics are reported
  as the mean (with min-max) across seeds, so one lucky seed can't carry it.

Trading rules (fixed before any result was seen -- see backtest.engine):
signal at a bar's close, entry at the next bar's open, hold 5 bars, one
position at a time, 100% of equity per trade (no leverage), longs and shorts,
0.05% cost per side, trade only when the predicted BUY/SELL probability is
at least 0.5. Do not tune these on this period; that would turn the backtest
into an optimisation and its numbers into an overestimate.

Caveat on the period: 2025-05-30 onward is the partition the one-time test
check used. Reporting a backtest on it is evaluation, not selection -- but
any decision made because of these numbers (changing rules, features, or
thresholds) must be validated on data after 2026-10-01.

Usage:
    python scripts/backtest_walk_forward.py
    python scripts/backtest_walk_forward.py --seeds 5 --jobs -1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from backtest.engine import BacktestConfig, buy_and_hold, run_backtest
from features.pipeline import build_feature_dataset
from features.scaling import apply_scaler, fit_scaler
from ml.dataset import FEATURE_COLUMNS, separate_features_and_target
from ml.models.random_forest import RandomForestModel
from walk_forward_sweep import HORIZON, RF_PARAMS, THRESHOLD, build_combined_dataset, load_ohlcv

PRIMARY_CSV = PROJECT_ROOT / "data/raw/XAUUSD_1h.csv"
SECONDARY_CSVS = {"EURUSD": PROJECT_ROOT / "data/raw/EURUSD_1h.csv", "USDJPY": PROJECT_ROOT / "data/raw/USDJPY_1h.csv"}
START = "2025-04-28"
RETRAIN_EVERY_BARS = 500
CONFIG = BacktestConfig(
    initial_capital=10_000.0,
    position_fraction=1.0,
    min_confidence=0.5,
    holding_bars=HORIZON,
    cost_pct_per_side=0.0005,
    allow_short=True,
)
REPORT_DIR = PROJECT_ROOT / "reports"

SUMMARY_METRICS = [
    ("total_return", "Total return", "{:+.1%}"),
    ("annualized_return", "Annualised return", "{:+.1%}"),
    ("sharpe_ratio", "Sharpe (annualised)", "{:+.2f}"),
    ("max_drawdown", "Max drawdown", "{:.1%}"),
    ("n_trades", "Trades", "{:.0f}"),
    ("win_rate", "Win rate", "{:.1%}"),
    ("profit_factor", "Profit factor", "{:.2f}"),
    ("avg_trade_return", "Avg trade return", "{:+.3%}"),
    ("exposure", "Time in market", "{:.1%}"),
    ("total_costs", "Costs paid ($)", "{:,.0f}"),
]


def predict_block(dataset: pd.DataFrame, columns: list[str], start: int, end: int, seed: int) -> pd.DataFrame:
    """Train on rows before `start` (minus the label-horizon purge), predict rows start..end."""
    train = dataset.iloc[: start - HORIZON]
    block = dataset.iloc[start:end]
    X_train, y_train = separate_features_and_target(train, feature_columns=columns)
    scaler = fit_scaler(X_train)
    model = RandomForestModel(random_state=seed, **RF_PARAMS)
    model.fit(apply_scaler(scaler, X_train), y_train)
    proba = model.predict_proba(apply_scaler(scaler, block[columns]))
    proba.index = block.index
    return proba


def walk_forward_signals(dataset, columns, start_position, seed, jobs) -> pd.DataFrame:
    starts = range(start_position, len(dataset), RETRAIN_EVERY_BARS)
    blocks = Parallel(n_jobs=jobs)(
        delayed(predict_block)(dataset, columns, s, min(s + RETRAIN_EVERY_BARS, len(dataset)), seed) for s in starts
    )
    return pd.concat(blocks)


def summarise(runs: list[dict]) -> dict:
    return {
        key: {"mean": float(np.mean([r[key] for r in runs])),
              "min": float(np.min([r[key] for r in runs])),
              "max": float(np.max([r[key] for r in runs]))}
        for key, _, _ in SUMMARY_METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()

    primary = load_ohlcv(PRIMARY_CSV)
    secondary_frames = {name: load_ohlcv(path) for name, path in SECONDARY_CSVS.items()}
    base_ds = build_feature_dataset(primary, horizon=HORIZON, threshold=THRESHOLD)
    combined_ds, cross_columns = build_combined_dataset(base_ds, primary, secondary_frames)
    common = base_ds.index.intersection(combined_ds.index).sort_values()
    configs = {
        "base": (base_ds.loc[common], FEATURE_COLUMNS),
        "plus_combined": (combined_ds.loc[common], FEATURE_COLUMNS + cross_columns),
    }
    start_position = int(common.searchsorted(pd.Timestamp(START, tz="UTC")))
    n_blocks = -(-(len(common) - start_position) // RETRAIN_EVERY_BARS)
    print(f"Backtest {common[start_position]} to {common[-1]}: {len(common) - start_position} bars, "
          f"retrained {n_blocks} times per run, {args.seeds} seeds per configuration.")
    print(f"Rules (fixed in advance): {CONFIG}\n")

    results: dict[str, list] = {}
    for name, (dataset, columns) in configs.items():
        results[name] = []
        for seed in range(args.seeds):
            signals = walk_forward_signals(dataset, columns, start_position, seed, args.jobs)
            results[name].append(run_backtest(primary, signals, CONFIG))
        print(f"done {name}")

    first_signal = common[start_position]
    # Align the benchmark to the strategies' last bar so all cover the same window.
    last_bar = results["base"][0].equity.index[-1]
    benchmark = buy_and_hold(primary.loc[:last_bar], first_signal, CONFIG)

    summaries = {name: summarise([r.metrics for r in runs]) for name, runs in results.items()}
    summaries["buy_and_hold"] = {key: {"mean": benchmark.metrics[key]} for key, _, _ in SUMMARY_METRICS}

    print(f"\n=== Out-of-sample backtest, {first_signal:%Y-%m-%d} to {last_bar:%Y-%m-%d} "
          f"(mean over {args.seeds} seeds, [min to max]) ===")
    print(f"{'':<22}{'base':>26}{'plus_combined':>26}{'buy & hold':>14}")
    for key, label, fmt in SUMMARY_METRICS:
        cells = []
        for name in ("base", "plus_combined"):
            s = summaries[name][key]
            cells.append(f"{fmt.format(s['mean'])} [{fmt.format(s['min'])} to {fmt.format(s['max'])}]")
        bh = summaries["buy_and_hold"][key]["mean"]
        bh_cell = fmt.format(bh) if key in ("total_return", "annualized_return", "sharpe_ratio", "max_drawdown") else "-"
        print(f"{label:<22}{cells[0]:>26}{cells[1]:>26}{bh_cell:>14}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    equity = pd.DataFrame({
        "base_seed0": results["base"][0].equity,
        "plus_combined_seed0": results["plus_combined"][0].equity,
        "buy_and_hold": benchmark.equity,
    })
    equity.to_csv(REPORT_DIR / "backtest_equity.csv", index_label="timestamp")
    results["plus_combined"][0].trades.to_csv(REPORT_DIR / "backtest_trades_plus_combined_seed0.csv", index=False)
    (REPORT_DIR / "backtest_walk_forward.json").write_text(json.dumps({
        "period": [str(first_signal), str(last_bar)],
        "rules": CONFIG.__dict__,
        "retrain_every_bars": RETRAIN_EVERY_BARS,
        "seeds": args.seeds,
        "summary": summaries,
    }, indent=2, default=str))
    print(f"\nSaved reports/backtest_walk_forward.json, reports/backtest_equity.csv, "
          f"reports/backtest_trades_plus_combined_seed0.csv")


if __name__ == "__main__":
    main()
