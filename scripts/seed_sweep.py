"""Multi-seed comparison of feature sets, validation-only.

Why this exists: single-run validation macro-F1 differences of 0.002-0.02
between feature sets have flipped between runs (see README's cross-asset
note), so one run cannot settle whether a feature set helps. This script
fits the same Random Forest hyperparameters across many random seeds per
feature set and reports mean +/- std of VALIDATION macro-F1, plus a paired
per-seed difference against `base`.

It never touches the test partition. Hyperparameters are fixed (the ones
the last full tuning run selected for `base`) rather than re-tuned per
seed, so the only thing varying between runs of one feature set is the
forest's own randomness -- that isolates the noise that made single runs
unreliable. It does not measure data-split noise; treat a gap as real only
if it is large relative to the std AND consistent in sign across seeds.

Usage:
    python scripts/seed_sweep.py
    python scripts/seed_sweep.py --secondary DXY=data/raw/DXY_1h.csv
    python scripts/seed_sweep.py --seeds 20 --secondary DXY=data/raw/DXY_1h.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from features.pipeline import build_feature_dataset
from features.scaling import apply_scaler, fit_scaler
from ml.dataset import FEATURE_COLUMNS, chronological_split, separate_features_and_target
from ml.evaluation import classification_report
from ml.models.random_forest import RandomForestModel

HORIZON = 5
THRESHOLD = 0.005
TRAIN_PCT = 0.70
VAL_PCT = 0.15
RF_PARAMS = {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 5, "class_weight": "balanced"}
MOMENTUM_COLUMNS = ["return_3", "return_10", "return_20"]
CROSS_ASSET_COLUMNS = ["secondary_log_return", "ratio_log_return", "rolling_correlation"]
CORR_WINDOW = 20


def load_ohlcv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    return raw.set_index("timestamp")


def val_macro_f1(train: pd.DataFrame, val: pd.DataFrame, columns: list[str], seed: int) -> float:
    X_train, y_train = separate_features_and_target(train, feature_columns=columns)
    X_val, y_val = separate_features_and_target(val, feature_columns=columns)
    scaler = fit_scaler(X_train)
    model = RandomForestModel(random_state=seed, **RF_PARAMS)
    model.fit(apply_scaler(scaler, X_train), y_train)
    Xv = apply_scaler(scaler, X_val)
    report = classification_report(y_val, model.predict(Xv), model.predict_proba(Xv))
    return report["macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--primary", default=str(PROJECT_ROOT / "data/raw/XAUUSD_1h.csv"))
    parser.add_argument(
        "--secondary", action="append", default=[],
        help="NAME=path.csv of another instrument's 1h OHLCV; repeatable. "
             "Default is EURUSD=data/raw/EURUSD_1h.csv when nothing is given.",
    )
    args = parser.parse_args()

    secondaries = dict(item.split("=", 1) for item in args.secondary)
    if not secondaries:
        secondaries = {"EURUSD": str(PROJECT_ROOT / "data/raw/EURUSD_1h.csv")}

    primary = load_ohlcv(Path(args.primary))
    seeds = list(range(args.seeds))

    # (name, columns, (train, val)). Single-symbol sets share one dataset
    # and split; each cross-asset set gets its own, because cross-asset
    # alignment NaNs would otherwise shrink the shared sample (same reason
    # train_xauusd_model.py keeps them separate).
    base_ds = build_feature_dataset(primary, horizon=HORIZON, threshold=THRESHOLD)
    base_train, base_val, _ = chronological_split(base_ds, horizon=HORIZON, train_pct=TRAIN_PCT, val_pct=VAL_PCT)
    sets = [
        ("base", FEATURE_COLUMNS, (base_train, base_val)),
        ("plus_momentum", FEATURE_COLUMNS + MOMENTUM_COLUMNS, (base_train, base_val)),
    ]
    for name, path in secondaries.items():
        ds = build_feature_dataset(
            primary, horizon=HORIZON, threshold=THRESHOLD,
            secondary_dataframe=load_ohlcv(Path(path)), cross_asset_corr_window=CORR_WINDOW,
        )
        tr, va, _ = chronological_split(ds, horizon=HORIZON, train_pct=TRAIN_PCT, val_pct=VAL_PCT)
        sets.append((f"plus_{name.lower()}", FEATURE_COLUMNS + CROSS_ASSET_COLUMNS, (tr, va)))
        print(f"{name}: {len(ds)} rows with cross-asset features (vs {len(base_ds)} base)")

    scores: dict[str, np.ndarray] = {}
    for name, columns, (tr, va) in sets:
        scores[name] = np.array([val_macro_f1(tr, va, columns, seed) for seed in seeds])
        print(f"done {name}")

    print(f"\n=== Validation macro-F1 over {len(seeds)} seeds (fixed RF params, test untouched) ===")
    print(f"{'feature set':<22}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}{'vs base':>10}{'seeds > base':>14}")
    base_scores = scores["base"]
    for name, s in scores.items():
        diff = s - base_scores
        wins = "-" if name == "base" else f"{int((diff > 0).sum())}/{len(seeds)}"
        delta = "-" if name == "base" else f"{diff.mean():+.4f}"
        print(f"{name:<22}{s.mean():>8.4f}{s.std(ddof=1):>8.4f}{s.min():>8.4f}{s.max():>8.4f}{delta:>10}{wins:>14}")

    print(
        "\nRead this as: a feature set is only a credible improvement if its mean gap vs base "
        "is several times the seed std AND it beats base on nearly every seed."
    )


if __name__ == "__main__":
    main()