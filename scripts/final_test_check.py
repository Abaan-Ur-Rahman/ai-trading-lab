"""One-time, pre-registered test-set check: base vs base + EUR/USD + USD/JPY.

Why this exists: walk_forward_sweep.py found that adding EUR/USD and USD/JPY
cross-asset features together (`plus_combined`) beat `base` in 5/5
validation periods (+0.022 mean macro-F1). But that winner was picked from
5 candidates scored on the same 5 periods, which makes its score look a bit
better than it really is. The untouched test partition is the honest check.

What is fixed BEFORE looking (do not change these after running for real):
- Exactly two configurations: `base` and `plus_combined`. No other set is
  scored on test, so the test result cannot be used to pick a winner.
- Fixed Random Forest hyperparameters (the same ones every sweep used),
  trained on all development rows (train + validation), averaged over
  SEEDS forest seeds.
- Same rows for both (the intersection of their datasets), and the same
  test partition chronological_split / train_xauusd_model.py use.
- Decision rule, stated in DECISION_RULE below and printed before results.

Running for real writes reports/final_test_check.json and refuses to run
again while that file exists, so "one-time" is enforced rather than just
intended. Commit that file: it is the record of the result.

--dry-run runs the identical code with the old validation partition standing
in for test (training on the train partition only). Use it to check the
script works; it never touches test data and writes nothing.

Usage:
    python scripts/final_test_check.py --dry-run
    python scripts/final_test_check.py          # the one-time real run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd

from features.pipeline import build_feature_dataset
from features.scaling import apply_scaler, fit_scaler
from ml.dataset import FEATURE_COLUMNS, separate_features_and_target
from ml.evaluation import classification_report, trading_report
from ml.models.random_forest import RandomForestModel
from ml.persistence import current_library_versions
from walk_forward_sweep import (
    HORIZON,
    RF_PARAMS,
    THRESHOLD,
    TRAIN_PCT,
    VAL_PCT,
    build_combined_dataset,
    load_ohlcv,
)

SEEDS = list(range(10))
MIN_CONFIDENCE = 0.5
TRANSACTION_COST_PCT = 0.0005
PRIMARY_PATH = PROJECT_ROOT / "data/raw/XAUUSD_1h.csv"
SECONDARY_PATHS = {
    "EURUSD": PROJECT_ROOT / "data/raw/EURUSD_1h.csv",
    "USDJPY": PROJECT_ROOT / "data/raw/USDJPY_1h.csv",
}
RESULT_PATH = PROJECT_ROOT / "reports/final_test_check.json"
DECISION_RULE = (
    "Dollar features count as confirmed on test only if plus_combined has a higher "
    "mean test macro-F1 than base AND beats base on at least 8 of 10 seeds. "
    "Trading metrics are reported as signal-quality diagnostics only and are not "
    "part of the decision."
)


def evaluate(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int) -> dict:
    X_train, y_train = separate_features_and_target(train, feature_columns=columns)
    X_test, y_test = separate_features_and_target(test, feature_columns=columns)
    scaler = fit_scaler(X_train)
    model = RandomForestModel(random_state=seed, **RF_PARAMS)
    model.fit(apply_scaler(scaler, X_train), y_train)
    X_test_scaled = apply_scaler(scaler, X_test)
    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)
    classification = classification_report(y_test, pred, proba)
    trading = trading_report(
        pred, proba, test, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    return {
        "macro_f1": classification["macro_f1"],
        "accuracy": classification["accuracy"],
        "per_class_f1": {name: c["f1"] for name, c in classification["per_class"].items()},
        "cumulative_return": trading["cumulative_return"],
        "hit_rate": trading["hit_rate"],
        "n_trades_taken": trading["n_trades_taken"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="use the old validation partition as a stand-in for test; touches no test data")
    args = parser.parse_args()

    if not args.dry_run and RESULT_PATH.exists():
        sys.exit(
            f"{RESULT_PATH} already exists: the one-time test check has been run. "
            "Re-running it would turn the test set into a tuning set. Read that file instead."
        )

    primary = load_ohlcv(PRIMARY_PATH)
    secondary_frames = {name: load_ohlcv(path) for name, path in SECONDARY_PATHS.items()}

    base_ds = build_feature_dataset(primary, horizon=HORIZON, threshold=THRESHOLD)
    combined_ds, combined_columns = build_combined_dataset(base_ds, primary, secondary_frames)
    configs = {
        "base": (FEATURE_COLUMNS, base_ds),
        "plus_combined": (FEATURE_COLUMNS + combined_columns, combined_ds),
    }

    common = base_ds.index.intersection(combined_ds.index).sort_values()
    n = len(common)
    train_end = int(n * TRAIN_PCT)
    dev_end = train_end + int(n * VAL_PCT)
    if args.dry_run:
        # Stand-in: train on the train partition, "test" on the old validation partition.
        train_index, test_index = common[:train_end], common[train_end + HORIZON:dev_end]
        label = "DRY RUN -- old validation partition standing in for test (no test data used)"
    else:
        # Real: train on all development rows, test on the untouched partition.
        train_index, test_index = common[:dev_end], common[dev_end + HORIZON:]
        label = "REAL ONE-TIME TEST CHECK"

    print(f"=== {label} ===")
    print(f"Training rows: {len(train_index)} ({train_index[0]} to {train_index[-1]})")
    print(f"Evaluated rows: {len(test_index)} ({test_index[0]} to {test_index[-1]})")
    print(f"Decision rule (fixed before looking): {DECISION_RULE}\n")

    results: dict[str, list[dict]] = {}
    for name, (columns, ds) in configs.items():
        train, test = ds.loc[train_index], ds.loc[test_index]
        results[name] = [evaluate(train, test, columns, seed) for seed in SEEDS]
        print(f"done {name} ({len(columns)} features)")

    f1 = {name: np.array([r["macro_f1"] for r in runs]) for name, runs in results.items()}
    gap = f1["plus_combined"] - f1["base"]
    wins = int((gap > 0).sum())
    confirmed = bool(gap.mean() > 0 and wins >= 8)

    def mean_of(name: str, key: str) -> float:
        return float(np.mean([r[key] for r in results[name]]))

    print(f"\n=== Macro-F1 over {len(SEEDS)} seeds ===")
    print(f"{'config':<16}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    for name, scores in f1.items():
        print(f"{name:<16}{scores.mean():>8.4f}{scores.std(ddof=1):>8.4f}{scores.min():>8.4f}{scores.max():>8.4f}")
    print(f"\nplus_combined vs base: mean gap {gap.mean():+.4f}, beats base on {wins}/{len(SEEDS)} seeds")
    print(f"Decision: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'} under the pre-registered rule.")

    print("\n=== Signal-quality diagnostics (seed means; not a portfolio backtest, not part of the decision) ===")
    print(f"{'config':<16}{'accuracy':>10}{'SELL F1':>9}{'HOLD F1':>9}{'BUY F1':>9}{'cum. return':>13}{'hit rate':>10}{'trades':>8}")
    for name, runs in results.items():
        per_class = {cls: float(np.mean([r["per_class_f1"][cls] for r in runs])) for cls in ("SELL", "HOLD", "BUY")}
        print(
            f"{name:<16}{mean_of(name, 'accuracy'):>10.4f}{per_class['SELL']:>9.4f}{per_class['HOLD']:>9.4f}"
            f"{per_class['BUY']:>9.4f}{mean_of(name, 'cumulative_return'):>+13.4f}{mean_of(name, 'hit_rate'):>10.4f}"
            f"{mean_of(name, 'n_trades_taken'):>8.0f}"
        )

    if args.dry_run:
        print("\nDry run only: nothing written, test partition untouched.")
        return

    record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "decision_rule": DECISION_RULE,
        "decision": "confirmed" if confirmed else "not_confirmed",
        "macro_f1_mean_gap": float(gap.mean()),
        "seeds_beating_base": wins,
        "configs": {
            name: {"feature_columns": configs[name][0], "macro_f1_by_seed": f1[name].tolist(), "runs": runs}
            for name, runs in results.items()
        },
        "train_rows": [str(train_index[0]), str(train_index[-1]), len(train_index)],
        "test_rows": [str(test_index[0]), str(test_index[-1]), len(test_index)],
        "rf_params": RF_PARAMS,
        "seeds": SEEDS,
        "horizon": HORIZON,
        "threshold": THRESHOLD,
        "library_versions": current_library_versions(),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(record, indent=2))
    print(f"\nRecorded to {RESULT_PATH}. Commit it; this check is now used up.")


if __name__ == "__main__":
    main()