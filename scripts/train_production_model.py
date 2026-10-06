"""Fit and save the validated production model (gold + EUR/USD + USD/JPY features).

What this model is, and why this configuration:
- Feature set `plus_combined`: the 5 base indicators plus 3 cross-asset
  features each for EUR/USD and USD/JPY (11 features). It beat `base` in 5/5
  walk-forward periods (scripts/walk_forward_sweep.py) and then passed a
  pre-registered one-time test check (scripts/final_test_check.py,
  reports/final_test_check.json): 0.3831 vs 0.3678 macro-F1, 10/10 seeds.
- Random Forest with the same fixed hyperparameters every one of those checks
  used. Hyperparameters are not re-tuned here: the checks showed tuning noise
  was larger than the effect being measured, and re-tuning would produce a
  model that was never actually validated.

Why it trains on ALL data, including the old test period: evaluation is
finished (that is what the walk-forward and the one-time test check were
for). A model that will be used going forward should learn from the most
recent data available; the dry run showed the dollar relationship helps most
when training data is recent. Consequently this model has NO held-out score
of its own -- the evidence for it is the checks above, recorded in its
metadata's evaluation_metrics, and any future evaluation must use data from
after its training window ends.

Does not touch scripts/train_xauusd_model.py, which remains the original
model-selection experiment.

Usage:
    python scripts/train_production_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from data.storage.csv_repository import CSVRepository
from features.pipeline import build_feature_dataset
from features.scaling import apply_scaler, fit_scaler
from ml.dataset import CLASS_LABELS, CLASS_NAMES, FEATURE_COLUMNS, separate_features_and_target
from ml.experiment_tracking import log_experiment
from ml.inference import predict_from_ohlcv
from ml.models.random_forest import RandomForestModel
from ml.persistence import ModelMetadata, current_library_versions, load_model, save_model
from walk_forward_sweep import CORR_WINDOW, HORIZON, RF_PARAMS, THRESHOLD, build_combined_dataset

SYMBOL = "XAU/USD"
TIMEFRAME = "1h"
PRIMARY_CSV = PROJECT_ROOT / "data/raw/XAUUSD_1h.csv"
# {column-prefix name: provider symbol}. Names must match the ones used in the
# validation checks, since they determine the feature column names.
SECONDARY_SYMBOLS = {"EURUSD": "EUR/USD", "USDJPY": "USD/JPY"}
TOTAL_CANDLES = 20_000
RANDOM_STATE = 42
MODEL_DIRECTORY = PROJECT_ROOT / "models/XAUUSD_1h_rf_balanced_plus_combined_production_v1"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "experiments/experiments.jsonl"
FINAL_TEST_CHECK_PATH = PROJECT_ROOT / "reports/final_test_check.json"
NOT_APPLICABLE = "n/a: production fit on all data; see evaluation_metrics for the validation evidence"


def load_or_fetch(symbol: str, csv_path: Path) -> pd.DataFrame:
    """Load a cached 1h OHLCV CSV, fetching it from Twelve Data only if missing."""
    if csv_path.exists():
        raw = pd.read_csv(csv_path)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        return raw.set_index("timestamp")

    # Imported lazily: the provider needs an API key, which a fully cached run does not.
    from data.providers.twelve_data import TwelveDataProvider
    from train_xauusd_model import load_or_fetch_raw_ohlcv

    return load_or_fetch_raw_ohlcv(
        CSVRepository(), TwelveDataProvider(), symbol=symbol, csv_path=csv_path, total_candles=TOTAL_CANDLES,
    )


def validation_evidence() -> dict:
    """Summarise the checks that justified this configuration, for the model's metadata."""
    evidence: dict = {
        "selection": (
            "walk_forward_sweep.py, 5 expanding-window periods 2025-04-28 to 2026-05-29: "
            "plus_combined beat base in 5/5 periods, mean macro-F1 gap +0.0218"
        ),
    }
    if FINAL_TEST_CHECK_PATH.exists():
        record = json.loads(FINAL_TEST_CHECK_PATH.read_text())
        evidence["final_test_check"] = {
            "decision": record["decision"],
            "decision_rule": record["decision_rule"],
            "macro_f1_mean_gap": record["macro_f1_mean_gap"],
            "seeds_beating_base": record["seeds_beating_base"],
            "test_rows": record["test_rows"],
            "source": "reports/final_test_check.json",
        }
    else:
        evidence["final_test_check"] = "reports/final_test_check.json not found in this checkout"
    return evidence


def main() -> None:
    primary = load_or_fetch(SYMBOL, PRIMARY_CSV)
    secondary_frames = {
        name: load_or_fetch(symbol, PROJECT_ROOT / f"data/raw/{name}_1h.csv")
        for name, symbol in SECONDARY_SYMBOLS.items()
    }

    base_ds = build_feature_dataset(primary, horizon=HORIZON, threshold=THRESHOLD)
    dataset, cross_asset_columns = build_combined_dataset(base_ds, primary, secondary_frames)
    feature_columns = FEATURE_COLUMNS + cross_asset_columns
    print(f"Training on all {len(dataset)} rows ({dataset.index.min()} to {dataset.index.max()}), "
          f"{len(feature_columns)} features.")

    X, y = separate_features_and_target(dataset, feature_columns=feature_columns)
    scaler = fit_scaler(X)
    model = RandomForestModel(random_state=RANDOM_STATE, **RF_PARAMS)
    model.fit(apply_scaler(scaler, X), y)

    counts = y.value_counts()
    metadata = ModelMetadata(
        model_type="RandomForestModel",
        hyperparameters=model.get_hyperparameters(),
        feature_columns=feature_columns,
        random_state=RANDOM_STATE,
        train_pct=1.0,
        val_pct=0.0,
        horizon=HORIZON,
        threshold=THRESHOLD,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        train_start=str(dataset.index.min()),
        train_end=str(dataset.index.max()),
        val_start=NOT_APPLICABLE,
        val_end=NOT_APPLICABLE,
        test_start=NOT_APPLICABLE,
        test_end=NOT_APPLICABLE,
        class_distribution={name: int(counts.get(label, 0)) for label, name in zip(CLASS_LABELS, CLASS_NAMES)},
        evaluation_metrics={**validation_evidence(), "feature_importances": model.get_feature_importances()},
        secondary_symbols=SECONDARY_SYMBOLS,
        cross_asset_corr_window=CORR_WINDOW,
        **current_library_versions(),
    )
    save_model(MODEL_DIRECTORY, model, scaler, metadata)
    print(f"Saved to {MODEL_DIRECTORY}")

    log_experiment(
        EXPERIMENT_LOG_PATH, metadata, MODEL_DIRECTORY,
        notes=(
            "Production fit of the validated plus_combined configuration (base + EUR/USD + USD/JPY "
            "cross-asset features, fixed RF hyperparameters) on all available data. No held-out "
            "score of its own; validation evidence is in evaluation_metrics."
        ),
    )
    print(f"Logged to {EXPERIMENT_LOG_PATH}")

    # Smoke check: reload exactly as a consumer would and predict the latest bar,
    # driving everything (columns, secondaries, window) from the saved metadata.
    loaded_model, loaded_scaler, loaded = load_model(MODEL_DIRECTORY)
    proba = predict_from_ohlcv(
        primary, loaded_model, loaded_scaler,
        feature_columns=loaded.feature_columns,
        secondary_ohlcv={name: secondary_frames[name] for name in loaded.secondary_symbols},
        cross_asset_corr_window=loaded.cross_asset_corr_window,
    )
    print(f"\nReload + live-style prediction for the latest bar ({primary.index.max()}):")
    for name, value in proba.items():
        print(f"  {name:<5} {value:.3f}")

    print("\nTop feature importances:")
    importances = sorted(model.get_feature_importances().items(), key=lambda item: item[1], reverse=True)
    for name, value in importances[:6]:
        print(f"  {name:<32} {value:.3f}")


if __name__ == "__main__":
    main()
