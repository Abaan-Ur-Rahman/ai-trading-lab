"""End-to-end training script: fetch XAUUSD 1h data, train, evaluate, persist.

This is orchestration, not a src/ layer module: every function it calls
(data fetching, feature building, splitting, scaling, training, baselines,
evaluation, persistence, experiment logging) already has its own unit
tests. This script just wires them together in the right order and is not
itself unit tested, the same way a thin CLI entry point normally isn't --
the logic worth testing lives one level down, where it's already covered.

Model selection: every candidate model is trained and evaluated on
VALIDATION only. Whichever has the higher validation macro-F1 is the
winner, decided here in code -- not assumed or hardcoded -- and only the
winner gets the one-time, final TEST-set check and gets persisted. The
loser is never evaluated on test at all, since repeatedly looking at test
across candidates is the same test-set leakage we're avoiding by not using
test for selection in the first place.

Run with: python scripts/train_xauusd_model.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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
from ml.models.base import ModelWrapper
from ml.models.logistic_regression import LogisticRegressionModel
from ml.models.random_forest import RandomForestModel
from ml.persistence import ModelMetadata, current_library_versions, save_model
from ml.training import train_model

SYMBOL = "XAU/USD"
SYMBOL_FOR_FILENAMES = "XAUUSD"
TIMEFRAME = "1h"
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / f"{SYMBOL_FOR_FILENAMES}_{TIMEFRAME}.csv"
EXPERIMENT_LOG_PATH = PROJECT_ROOT / "experiments" / "experiments.jsonl"
MODELS_DIR = PROJECT_ROOT / "models"

TOTAL_CANDLES = 20_000  # ~2.3 years of 1h XAUUSD data, via paginated fetch
HORIZON = 5
THRESHOLD = 0.005
TRAIN_PCT = 0.70
VAL_PCT = 0.15
RANDOM_STATE = 42
MIN_CONFIDENCE = 0.5
TRANSACTION_COST_PCT = 0.0005

LOGREG_CLASS_WEIGHT = "balanced"
RF_CLASS_WEIGHT = "balanced"


@dataclass
class Candidate:
    """One trained model competing for selection, plus its validation score."""

    name: str
    model: ModelWrapper
    model_directory: Path
    val_macro_f1: float
    val_classification: dict
    val_trading: dict


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

    print(f"No cached data found. Fetching {TOTAL_CANDLES} candles of {SYMBOL} {TIMEFRAME} "
          f"(paginated, ~{-(-TOTAL_CANDLES // 5000)} requests)...")
    service = MarketDataService(provider)
    candles = service.get_historical_candles(symbol=SYMBOL, timeframe=TIMEFRAME, total_candles=TOTAL_CANDLES)
    raw = candles_to_dataframe(candles)
    print(f"Fetched {len(raw)} candles.")

    repository.save(raw.reset_index(), RAW_CSV_PATH)
    print(f"Cached raw data to {RAW_CSV_PATH}")

    return raw


def build_class_distribution(y: pd.Series) -> dict[str, int]:
    """Map a class-label Series' value counts onto CLASS_NAMES, zero-filled."""
    counts = y.value_counts()
    return {name: int(counts.get(label, 0)) for label, name in zip(CLASS_LABELS, CLASS_NAMES)}


def model_directory_for(name: str, class_weight: str | None) -> Path:
    """Build this model's persistence directory, tagged by its class_weight config."""
    tag = class_weight if class_weight is not None else "none"
    return MODELS_DIR / f"{SYMBOL_FOR_FILENAMES}_{TIMEFRAME}_{name}_{tag}_v1"


def evaluate_candidate(
    name: str,
    model: ModelWrapper,
    model_directory: Path,
    X_val_scaled: pd.DataFrame,
    y_val: pd.Series,
    val_ohlcv: pd.DataFrame,
) -> Candidate:
    """Run this fitted model's VALIDATION-only evaluation and print it.

    Never touches test -- that's reserved for whichever candidate wins.
    """
    val_pred = model.predict(X_val_scaled)
    val_proba = model.predict_proba(X_val_scaled)

    val_classification = classification_report(y_val, val_pred, val_proba)
    val_trading = trading_report(
        val_pred, val_proba, val_ohlcv, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    )

    print(f"\n=== {name} -- classification report (VALIDATION set) ===")
    print(val_classification)
    print(f"\n=== {name} -- trading report (VALIDATION set) ===")
    print(val_trading)

    return Candidate(
        name=name,
        model=model,
        model_directory=model_directory,
        val_macro_f1=val_classification["macro_f1"],
        val_classification=val_classification,
        val_trading=val_trading,
    )


def main(provider=None) -> None:
    provider = provider or TwelveDataProvider()
    repository = CSVRepository()

    raw_ohlcv = load_or_fetch_raw_ohlcv(repository, provider)

    print("Building feature dataset...")
    dataset = build_feature_dataset(raw_ohlcv, horizon=HORIZON, threshold=THRESHOLD)
    print(f"Feature dataset: {len(dataset)} rows after warm-up/horizon trimming.")

    # Both candidates are trained against the identical split/scaler, produced
    # once here, so the comparison between them is apples-to-apples.
    logreg_result = train_model(
        LogisticRegressionModel(class_weight=LOGREG_CLASS_WEIGHT, random_state=RANDOM_STATE),
        dataset, horizon=HORIZON, train_pct=TRAIN_PCT, val_pct=VAL_PCT,
    )

    X_train, y_train = separate_features_and_target(logreg_result.train)
    X_val, y_val = separate_features_and_target(logreg_result.val)
    X_test, y_test = separate_features_and_target(logreg_result.test)
    X_train_scaled = apply_scaler(logreg_result.scaler, X_train)
    X_val_scaled = apply_scaler(logreg_result.scaler, X_val)
    X_test_scaled = apply_scaler(logreg_result.scaler, X_test)

    rf_model = RandomForestModel(class_weight=RF_CLASS_WEIGHT, random_state=RANDOM_STATE)
    rf_model.fit(X_train_scaled, y_train)

    # --- Candidate comparison: VALIDATION only. Neither candidate has
    # touched test yet. ---
    candidates = [
        evaluate_candidate(
            "LogisticRegression", logreg_result.model,
            model_directory_for("logreg", LOGREG_CLASS_WEIGHT),
            X_val_scaled, y_val, logreg_result.val,
        ),
        evaluate_candidate(
            "RandomForest", rf_model,
            model_directory_for("rf", RF_CLASS_WEIGHT),
            X_val_scaled, y_val, logreg_result.val,
        ),
    ]

    print("\n=== Random Forest -- feature importances ===")
    print(rf_model.get_feature_importances())

    winner = max(candidates, key=lambda candidate: candidate.val_macro_f1)
    comparison_summary = ", ".join(
        f"{c.name}={c.val_macro_f1:.4f}" for c in candidates
    )
    print(f"\nModel selection (validation macro-F1): {comparison_summary}")
    print(f"Winner: {winner.name}")

    # --- Winner only: ONE final, one-time test-set check. The loser is
    # never evaluated on test. ---
    y_pred = winner.model.predict(X_test_scaled)
    y_proba = winner.model.predict_proba(X_test_scaled)

    print(f"\n=== {winner.name} -- classification report (TEST set, final check only) ===")
    test_classification = classification_report(y_test, y_pred, y_proba)
    print(test_classification)

    print(f"\n=== {winner.name} -- trading report (TEST set, final check only) ===")
    test_trading = trading_report(
        y_pred, y_proba, logreg_result.test, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    print(test_trading)

    # Baselines are a fixed reference floor, not candidates being chosen
    # between, so evaluating them on test carries none of the
    # repeated-peeking risk that touching test for both ML candidates would.
    majority_baseline = MajorityClassBaseline()
    majority_baseline.fit(X_train, y_train)
    majority_pred = majority_baseline.predict(X_test)
    majority_proba = majority_baseline.predict_proba(X_test)
    print("\n=== Majority-class baseline -- classification report (test set) ===")
    print(classification_report(y_test, majority_pred, majority_proba))

    rule_based_baseline = RuleBasedBaseline()
    rule_pred = rule_based_baseline.predict(logreg_result.test)
    rule_proba = rule_based_baseline.predict_proba(logreg_result.test)
    print("\n=== Rule-based baseline -- classification report (test set) ===")
    print(classification_report(y_test, rule_pred, rule_proba))

    evaluation_metrics = {"classification": test_classification, "trading": test_trading}
    if winner.name == "RandomForest":
        evaluation_metrics["feature_importances"] = rf_model.get_feature_importances()

    print(f"\nPersisting winning model ({winner.name})...")
    metadata = ModelMetadata(
        model_type=f"{winner.name}Model",
        hyperparameters=winner.model.get_hyperparameters(),
        feature_columns=list(X_train.columns),
        random_state=RANDOM_STATE,
        train_pct=TRAIN_PCT,
        val_pct=VAL_PCT,
        horizon=HORIZON,
        threshold=THRESHOLD,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        train_start=str(logreg_result.train.index.min()),
        train_end=str(logreg_result.train.index.max()),
        val_start=str(logreg_result.val.index.min()),
        val_end=str(logreg_result.val.index.max()),
        test_start=str(logreg_result.test.index.min()),
        test_end=str(logreg_result.test.index.max()),
        class_distribution=build_class_distribution(y_train),
        evaluation_metrics=evaluation_metrics,
        **current_library_versions(),
    )
    save_model(winner.model_directory, winner.model, logreg_result.scaler, metadata)
    print(f"Model saved to {winner.model_directory}")

    log_experiment(
        EXPERIMENT_LOG_PATH, metadata, winner.model_directory,
        notes=f"Selected by validation macro-F1 over {len(candidates)} candidates: {comparison_summary}",
    )
    print(f"Experiment logged to {EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()