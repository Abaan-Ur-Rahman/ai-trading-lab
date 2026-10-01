"""End-to-end training script: fetch XAUUSD 1h data, train, evaluate, persist.

This is orchestration, not a src/ layer module: every function it calls
(data fetching, feature building, splitting, scaling, training, baselines,
evaluation, persistence, experiment logging) already has its own unit
tests. This script just wires them together in the right order and is not
itself unit tested, the same way a thin CLI entry point normally isn't --
the logic worth testing lives one level down, where it's already covered.

Run with: python scripts/train_xauusd_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from data.dataframe import candles_to_dataframe
from data.providers.twelve_data import TwelveDataProvider
from data.services.market_data import MarketDataService
from data.storage.csv_repository import CSVRepository
from features.pipeline import build_feature_dataset
from features.scaling import apply_scaler
from ml.baseline import MajorityClassBaseline, RuleBasedBaseline
from ml.dataset import CLASS_LABELS, CLASS_NAMES, separate_features_and_target
from ml.evaluation import classification_report, trading_report
from ml.experiment_tracking import log_experiment
from ml.models.logistic_regression import LogisticRegressionModel
from ml.persistence import ModelMetadata, current_library_versions, save_model
from ml.training import train_model
from ml.models.random_forest import RandomForestModel

SYMBOL = "XAU/USD"
SYMBOL_FOR_FILENAMES = "XAUUSD"
TIMEFRAME = "1h"
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / f"{SYMBOL_FOR_FILENAMES}_{TIMEFRAME}.csv"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "experiments" / "experiments.jsonl"

FETCH_LIMIT = 5000
HORIZON = 5
THRESHOLD = 0.005
TRAIN_PCT = 0.70
VAL_PCT = 0.15
CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42
MIN_CONFIDENCE = 0.5
TRANSACTION_COST_PCT = 0.0005

_CLASS_WEIGHT_TAG = CLASS_WEIGHT if CLASS_WEIGHT is not None else "none"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / f"{SYMBOL_FOR_FILENAMES}_{TIMEFRAME}_logreg_{_CLASS_WEIGHT_TAG}_v1"
RF_CLASS_WEIGHT = "balanced"
_RF_CLASS_WEIGHT_TAG = RF_CLASS_WEIGHT if RF_CLASS_WEIGHT is not None else "none"
RF_MODEL_DIRECTORY = PROJECT_ROOT / "models" / f"{SYMBOL_FOR_FILENAMES}_{TIMEFRAME}_rf_{_RF_CLASS_WEIGHT_TAG}_v1"


def load_or_fetch_raw_ohlcv(repository: CSVRepository, provider) -> pd.DataFrame:
    """Load cached OHLCV data if present, otherwise fetch and cache it.

    CSVRepository.save() writes via to_csv(index=False), which would
    silently drop a DatetimeIndex -- so the timestamp is moved into an
    ordinary column before saving, and restored as the index after
    loading, on both the cache-hit and cache-miss paths.
    """
    if repository.exists(RAW_CSV_PATH):
        print(f"Loading cached OHLCV data from {RAW_CSV_PATH}")
        raw = repository.load(RAW_CSV_PATH)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        return raw.set_index("timestamp")

    print(f"No cached data found. Fetching {FETCH_LIMIT} candles of {SYMBOL} {TIMEFRAME}...")
    service = MarketDataService(provider)
    candles = service.get_recent_candles(symbol=SYMBOL, timeframe=TIMEFRAME, limit=FETCH_LIMIT)
    raw = candles_to_dataframe(candles)
    print(f"Fetched {len(raw)} candles.")

    repository.save(raw.reset_index(), RAW_CSV_PATH)
    print(f"Cached raw data to {RAW_CSV_PATH}")

    return raw


def build_class_distribution(y: pd.Series) -> dict[str, int]:
    """Map a class-label Series' value counts onto CLASS_NAMES, zero-filled."""
    counts = y.value_counts()
    return {name: int(counts.get(label, 0)) for label, name in zip(CLASS_LABELS, CLASS_NAMES)}


def main(provider=None) -> None:
    provider = provider or TwelveDataProvider()
    repository = CSVRepository()

    raw_ohlcv = load_or_fetch_raw_ohlcv(repository, provider)

    print("Building feature dataset...")
    dataset = build_feature_dataset(raw_ohlcv, horizon=HORIZON, threshold=THRESHOLD)
    print(f"Feature dataset: {len(dataset)} rows after warm-up/horizon trimming.")

    model = LogisticRegressionModel(class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE)
    result = train_model(model, dataset, horizon=HORIZON, train_pct=TRAIN_PCT, val_pct=VAL_PCT)

    X_train, y_train = separate_features_and_target(result.train)
    X_val, y_val = separate_features_and_target(result.val)
    X_test, y_test = separate_features_and_target(result.test)
    X_train_scaled = apply_scaler(result.scaler, X_train)
    X_val_scaled = apply_scaler(result.scaler, X_val)
    X_test_scaled = apply_scaler(result.scaler, X_test)

    # VALIDATION set: use this to compare configurations (class_weight,
    # threshold, horizon, etc.). This is the set you're allowed to look at
    # repeatedly while iterating.
    val_pred = result.model.predict(X_val_scaled)
    val_proba = result.model.predict_proba(X_val_scaled)

    print("\n=== Logistic Regression -- classification report (VALIDATION set) ===")
    print(classification_report(y_val, val_pred, val_proba))

    print("\n=== Logistic Regression -- trading report (VALIDATION set) ===")
    print(trading_report(
        val_pred, val_proba, result.val, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    ))

    # TEST set: final, one-time check only. Do NOT use these numbers to
    # pick between configurations -- that's what validation is for. Only
    # look at this once you've already decided on your final setup.
    y_pred = result.model.predict(X_test_scaled)
    y_proba = result.model.predict_proba(X_test_scaled)

    print("\n=== Logistic Regression -- classification report (TEST set, final check only) ===")
    ml_classification = classification_report(y_test, y_pred, y_proba)
    print(ml_classification)

    print("\n=== Logistic Regression -- trading report (TEST set, final check only) ===")
    ml_trading = trading_report(
        y_pred, y_proba, result.test, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    print(ml_trading)

    # Baselines, evaluated on the exact same untouched test period.
    majority_baseline = MajorityClassBaseline()
    majority_baseline.fit(X_train, y_train)
    majority_pred = majority_baseline.predict(X_test)
    majority_proba = majority_baseline.predict_proba(X_test)
    print("\n=== Majority-class baseline -- classification report (test set) ===")
    print(classification_report(y_test, majority_pred, majority_proba))

    rule_based_baseline = RuleBasedBaseline()
    rule_pred = rule_based_baseline.predict(result.test)
    rule_proba = rule_based_baseline.predict_proba(result.test)
    print("\n=== Rule-based baseline -- classification report (test set) ===")
    print(classification_report(y_test, rule_pred, rule_proba))

    print("\nPersisting model...")
    metadata = ModelMetadata(
        model_type="LogisticRegressionModel",
        hyperparameters=result.model.get_hyperparameters(),
        feature_columns=list(X_train.columns),
        random_state=RANDOM_STATE,
        train_pct=TRAIN_PCT,
        val_pct=VAL_PCT,
        horizon=HORIZON,
        threshold=THRESHOLD,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        train_start=str(result.train.index.min()),
        train_end=str(result.train.index.max()),
        val_start=str(result.val.index.min()),
        val_end=str(result.val.index.max()),
        test_start=str(result.test.index.min()),
        test_end=str(result.test.index.max()),
        class_distribution=build_class_distribution(y_train),
        evaluation_metrics={"classification": ml_classification, "trading": ml_trading},
        **current_library_versions(),
    )
    save_model(MODEL_DIRECTORY, result.model, result.scaler, metadata)
    print(f"Model saved to {MODEL_DIRECTORY}")

    log_experiment(EXPERIMENT_LOG_PATH, metadata, MODEL_DIRECTORY, notes="First MVP end-to-end run")
    print(f"Experiment logged to {EXPERIMENT_LOG_PATH}")

    # --- Candidate comparison: Random Forest, validation set only ---
    # Not evaluated on test yet -- we're still choosing between this and
    # Logistic Regression. Whichever wins on validation gets ONE final
    # test-set run; this one doesn't touch test until that decision is made.
    rf_model = RandomForestModel(class_weight=RF_CLASS_WEIGHT, random_state=RANDOM_STATE)
    rf_model.fit(X_train_scaled, y_train)

    rf_val_pred = rf_model.predict(X_val_scaled)
    rf_val_proba = rf_model.predict_proba(X_val_scaled)

    print("\n=== Random Forest -- classification report (VALIDATION set) ===")
    print(classification_report(y_val, rf_val_pred, rf_val_proba))

    print("\n=== Random Forest -- trading report (VALIDATION set) ===")
    print(trading_report(
        rf_val_pred, rf_val_proba, result.val, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    ))

    print("\n=== Random Forest -- feature importances ===")
    print(rf_model.get_feature_importances())

    # Decision made on validation evidence: Random Forest outperforms
    # Logistic Regression (macro-F1 0.407 vs 0.367, more balanced per-class
    # recall). This is now the FINAL, one-time test-set check for the
    # chosen model -- not a comparison.
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)

    print("\n=== Random Forest -- classification report (TEST set, final check only) ===")
    rf_classification = classification_report(y_test, rf_pred, rf_proba)
    print(rf_classification)

    print("\n=== Random Forest -- trading report (TEST set, final check only) ===")
    rf_trading = trading_report(
        rf_pred, rf_proba, result.test, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    print(rf_trading)

    print("\nPersisting Random Forest model...")
    rf_metadata = ModelMetadata(
        model_type="RandomForestModel",
        hyperparameters=rf_model.get_hyperparameters(),
        feature_columns=list(X_train.columns),
        random_state=RANDOM_STATE,
        train_pct=TRAIN_PCT,
        val_pct=VAL_PCT,
        horizon=HORIZON,
        threshold=THRESHOLD,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        train_start=str(result.train.index.min()),
        train_end=str(result.train.index.max()),
        val_start=str(result.val.index.min()),
        val_end=str(result.val.index.max()),
        test_start=str(result.test.index.min()),
        test_end=str(result.test.index.max()),
        class_distribution=build_class_distribution(y_train),
        evaluation_metrics={
            "classification": rf_classification,
            "trading": rf_trading,
            "feature_importances": rf_model.get_feature_importances(),
        },
        **current_library_versions(),
    )
    save_model(RF_MODEL_DIRECTORY, rf_model, result.scaler, rf_metadata)
    print(f"Model saved to {RF_MODEL_DIRECTORY}")

    log_experiment(
        EXPERIMENT_LOG_PATH, rf_metadata, RF_MODEL_DIRECTORY,
        notes="Random Forest, chosen over Logistic Regression based on validation macro-F1 (0.407 vs 0.367)",
    )
    print(f"Experiment logged to {EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()