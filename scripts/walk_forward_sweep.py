"""Walk-forward comparison of feature sets across several time periods.

Why this exists: seed_sweep.py showed EUR/USD cross-asset features ahead of
`base` on average, but every seed was scored on the SAME validation window
(one ~4-month period). A feature that only helps in one regime would look
identical to one that helps generally. This script re-scores every feature
set on several consecutive later periods instead, so a real improvement has
to show up in most of them, not just one.

How it works:
- Every feature set is evaluated on exactly the same rows (the intersection
  of all datasets' timestamps). Cross-asset datasets lose a few rows to
  alignment/correlation warm-up; without this, sets would be scored on
  slightly different samples and the comparison would not be paired.
- Only the development portion (what train_xauusd_model.py calls train +
  validation) is used. The final test partition is cut off exactly as
  chronological_split would cut it, and never touched.
- Expanding-window folds: the first `--initial-train-pct` of development
  data is the first training window; the rest is split into `--folds`
  consecutive validation blocks. Each fold trains on everything before its
  block (minus a `horizon`-row purge gap, same convention as
  chronological_split) and scores macro-F1 on the block.
- Each (fold, feature set) is averaged over `--seeds` forest seeds, using the
  same fixed Random Forest hyperparameters as seed_sweep.py.

Read the result as: a feature set is a credible improvement only if it beats
`base` in most folds, not just on average.

Optional combined sets (both need 2+ secondaries):
- --combine adds `plus_combined`: every secondary's 3 cross-asset features
  side by side, prefixed by name (e.g. eurusd_rolling_correlation).
- --dollar-index adds `plus_dollar_index`: the 3 cross-asset features
  computed against an equal-weighted synthetic dollar index built from the
  EURUSD and USDJPY secondaries (see features.cross_asset.synthetic_dollar_index).

Usage:
    python scripts/walk_forward_sweep.py
    python scripts/walk_forward_sweep.py --folds 5 --seeds 5 \
        --secondary EURUSD=data/raw/EURUSD_1h.csv --secondary USDJPY=data/raw/USDJPY_1h.csv
    python scripts/walk_forward_sweep.py --folds 5 --seeds 5 --combine --dollar-index \
        --secondary EURUSD=data/raw/EURUSD_1h.csv --secondary USDJPY=data/raw/USDJPY_1h.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from features.cross_asset import build_cross_asset_features, synthetic_dollar_index
from features.pipeline import build_feature_dataset
from features.scaling import apply_scaler, fit_scaler
from ml.dataset import FEATURE_COLUMNS, separate_features_and_target
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


def build_combined_dataset(
    base_ds: pd.DataFrame,
    primary: pd.DataFrame,
    secondary_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """Base dataset plus every secondary's cross-asset features, prefixed by name.

    Each secondary's features are computed from the full primary close (same
    as build_feature_dataset does) and joined onto the base dataset's rows;
    only rows complete in every feature are kept. Returns the dataset and the
    added column names (e.g. eurusd_rolling_correlation, usdjpy_...).
    """
    combined = base_ds.copy()
    columns: list[str] = []
    for name, frame in secondary_frames.items():
        features = build_cross_asset_features(
            primary["close"], frame["close"], corr_window=CORR_WINDOW,
        ).add_prefix(f"{name.lower()}_")
        combined = combined.join(features)
        columns += list(features.columns)
    return combined.dropna(subset=columns), columns


def fold_bounds(n_dev: int, n_folds: int, initial_train_pct: float) -> list[tuple[int, int]]:
    """(val_start, val_end) positions for each expanding-window fold."""
    first_val_start = int(n_dev * initial_train_pct)
    block = (n_dev - first_val_start) // n_folds
    return [(first_val_start + k * block, first_val_start + (k + 1) * block) for k in range(n_folds)]


def score(train: pd.DataFrame, val: pd.DataFrame, columns: list[str], seed: int) -> float:
    X_train, y_train = separate_features_and_target(train, feature_columns=columns)
    X_val, y_val = separate_features_and_target(val, feature_columns=columns)
    scaler = fit_scaler(X_train)
    model = RandomForestModel(random_state=seed, **RF_PARAMS)
    model.fit(apply_scaler(scaler, X_train), y_train)
    Xv = apply_scaler(scaler, X_val)
    return classification_report(y_val, model.predict(Xv), model.predict_proba(Xv))["macro_f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--initial-train-pct", type=float, default=0.5)
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    parser.add_argument("--combine", action="store_true",
                        help="also test all secondaries' cross-asset features together (plus_combined)")
    parser.add_argument("--dollar-index", action="store_true",
                        help="also test a synthetic dollar index from the EURUSD and USDJPY secondaries")
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
    base_ds = build_feature_dataset(primary, horizon=HORIZON, threshold=THRESHOLD)
    sets: list[tuple[str, list[str], pd.DataFrame]] = [
        ("base", FEATURE_COLUMNS, base_ds),
        ("plus_momentum", FEATURE_COLUMNS + MOMENTUM_COLUMNS, base_ds),
    ]
    secondary_frames = {name: load_ohlcv(Path(path)) for name, path in secondaries.items()}
    for name, frame in secondary_frames.items():
        ds = build_feature_dataset(
            primary, horizon=HORIZON, threshold=THRESHOLD,
            secondary_dataframe=frame, cross_asset_corr_window=CORR_WINDOW,
        )
        sets.append((f"plus_{name.lower()}", FEATURE_COLUMNS + CROSS_ASSET_COLUMNS, ds))

    if args.combine:
        if len(secondary_frames) < 2:
            parser.error("--combine needs at least two --secondary instruments")
        combined, combined_columns = build_combined_dataset(base_ds, primary, secondary_frames)
        sets.append(("plus_combined", FEATURE_COLUMNS + combined_columns, combined))

    if args.dollar_index:
        if not {"EURUSD", "USDJPY"} <= set(secondary_frames):
            parser.error("--dollar-index needs secondaries named EURUSD and USDJPY")
        dollar = synthetic_dollar_index(secondary_frames["EURUSD"]["close"], secondary_frames["USDJPY"]["close"])
        ds = build_feature_dataset(
            primary, horizon=HORIZON, threshold=THRESHOLD,
            secondary_dataframe=dollar.to_frame(), cross_asset_corr_window=CORR_WINDOW,
        )
        sets.append(("plus_dollar_index", FEATURE_COLUMNS + CROSS_ASSET_COLUMNS, ds))

    common = sets[0][2].index
    for _, _, ds in sets[1:]:
        common = common.intersection(ds.index)
    common = common.sort_values()

    # Cut off the final test partition exactly where chronological_split would
    # (train_pct + val_pct of rows, then a horizon purge before test), and only
    # ever use what comes before it.
    n = len(common)
    dev_end = int(n * TRAIN_PCT) + int(n * VAL_PCT)
    dev_index = common[:dev_end]
    print(f"Common rows across all feature sets: {n}; development rows used: {len(dev_index)} "
          f"(test partition from {common[dev_end + HORIZON]} onward left untouched)")

    bounds = fold_bounds(len(dev_index), args.folds, args.initial_train_pct)
    seeds = list(range(args.seeds))

    tasks = []
    for fold, (vs, ve) in enumerate(bounds):
        for name, columns, ds in sets:
            dev = ds.loc[dev_index]
            train = dev.iloc[: vs - HORIZON]
            val = dev.iloc[vs:ve]
            for seed in seeds:
                tasks.append((fold, name, columns, train, val, seed))

    print(f"Fitting {len(tasks)} models ({args.folds} folds x {len(sets)} feature sets x {len(seeds)} seeds)...")
    results = Parallel(n_jobs=args.jobs)(
        delayed(score)(train, val, columns, seed) for _, _, columns, train, val, seed in tasks
    )

    frame = pd.DataFrame(
        [(fold, name, seed, f1) for (fold, name, _, _, _, seed), f1 in zip(tasks, results)],
        columns=["fold", "feature_set", "seed", "macro_f1"],
    )
    per_fold = frame.groupby(["fold", "feature_set"])["macro_f1"].mean().unstack()
    per_fold = per_fold[[name for name, _, _ in sets]]

    print("\n=== Mean validation macro-F1 per fold (averaged over seeds; test untouched) ===")
    header = f"{'fold':<6}{'validation period':<27}" + "".join(f"{name:>19}" for name, _, _ in sets)
    print(header)
    for fold, (vs, ve) in enumerate(bounds):
        period = f"{dev_index[vs]:%Y-%m-%d} to {dev_index[ve - 1]:%Y-%m-%d}"
        row = "".join(f"{per_fold.loc[fold, name]:>19.4f}" for name, _, _ in sets)
        print(f"{fold + 1:<6}{period:<27}{row}")

    print("\n=== Versus base, across folds ===")
    print(f"{'feature set':<18}{'mean gap':>10}{'gap std':>10}{'folds > base':>14}")
    for name, _, _ in sets[1:]:
        gap = per_fold[name] - per_fold["base"]
        print(f"{name:<18}{gap.mean():>+10.4f}{gap.std(ddof=1):>10.4f}{f'{int((gap > 0).sum())}/{len(gap)}':>14}")

    print(
        "\nRead this as: credible only if a feature set beats base in most folds "
        "(e.g. 4/5 or 5/5) with a mean gap clearly larger than the gap std."
    )


if __name__ == "__main__":
    main()