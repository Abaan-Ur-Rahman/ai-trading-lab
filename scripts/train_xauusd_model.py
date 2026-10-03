"""End-to-end training script: fetch XAUUSD 1h data, train, evaluate, persist.

This is orchestration, not a src/ layer module: every function it calls
(data fetching, feature building, splitting, scaling, training, baselines,
evaluation, persistence, experiment logging) already has its own unit
tests. This script just wires them together in the right order and is not
itself unit tested, the same way a thin CLI entry point normally isn't --
the logic worth testing lives one level down, where it's already covered.

Model selection: 4 feature-set configurations (base / plus_momentum /
plus_range / plus_momentum_range) x 2 models (LogisticRegression,
RandomForest) = 8 candidates, every one trained and evaluated on
VALIDATION only. Whichever has the higher validation macro-F1 is the
winner, decided here in code -- not assumed or hardcoded -- and only the
winner gets the one-time, final TEST-set check and gets persisted. No
other candidate is ever evaluated on test, since repeatedly looking at
test across candidates is the same test-set leakage we're avoiding by not
using test for selection in the first place.

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
from features.scaling import FeatureScaler, apply_scaler
from ml.baseline import MajorityClassBaseline, RuleBasedBaseline
from ml.dataset import CLASS_LABELS, CLASS_NAMES, FEATURE_COLUMNS, separate_features_and_target
from ml.evaluation import classification_report, trading_report
from ml.experiment_tracking import log_experiment
from ml.models.base import ModelWrapper
from ml.models.logistic_regression import LogisticRegressionModel
from ml.models.random_forest import RandomForestModel
from ml.persistence import ModelMetadata, current_library_versions, save_model
from ml.training import TrainingResult, train_model

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

# Feature-set names are constants, not strings built inline at each call
# site, so a typo in one place can't silently create a mismatched or
# duplicate configuration elsewhere (directory names, the summary table,
# and experiment notes all read from the same FeatureSet.name values).
FEATURE_SET_BASE = "base"
FEATURE_SET_PLUS_MOMENTUM = "plus_momentum"
FEATURE_SET_PLUS_RANGE = "plus_range"
FEATURE_SET_PLUS_MOMENTUM_RANGE = "plus_momentum_range"

MOMENTUM_COLUMNS = ["return_3", "return_10", "return_20"]
RANGE_COLUMNS = ["range_position"]


@dataclass
class FeatureSet:
    """One named feature-column configuration to compare against the others."""

    name: str
    columns: list[str]


FEATURE_SETS: list[FeatureSet] = [
    FeatureSet(FEATURE_SET_BASE, FEATURE_COLUMNS),
    FeatureSet(FEATURE_SET_PLUS_MOMENTUM, FEATURE_COLUMNS + MOMENTUM_COLUMNS),
    FeatureSet(FEATURE_SET_PLUS_RANGE, FEATURE_COLUMNS + RANGE_COLUMNS),
    FeatureSet(FEATURE_SET_PLUS_MOMENTUM_RANGE, FEATURE_COLUMNS + MOMENTUM_COLUMNS + RANGE_COLUMNS),
]


@dataclass
class Candidate:
    """One (feature set, model) combination competing for selection.

    Carries its own scaler because a scaler fit on one feature set's
    columns cannot be reused for a different feature set -- different
    column counts, different per-column statistics.
    """

    feature_set_name: str
    feature_columns: list[str]
    model_name: str
    model_tag: str
    class_weight: str | None
    model: ModelWrapper
    scaler: FeatureScaler
    model_directory: Path
    val_macro_f1: float
    val_log_loss: float
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


def model_directory_for(model_tag: str, class_weight: str | None, feature_set_name: str) -> Path:
    """Build this model's persistence directory, tagged by class_weight and feature set.

    Purely a human-readable label for browsing the models/ directory --
    the authoritative record of what a persisted model actually is lives
    in its metadata.json (feature_columns, hyperparameters, split config,
    etc.), not in this directory name.
    """
    class_weight_tag = class_weight if class_weight is not None else "none"
    return MODELS_DIR / (
        f"{SYMBOL_FOR_FILENAMES}_{TIMEFRAME}_{model_tag}_{class_weight_tag}_{feature_set_name}_v1"
    )


def evaluate_candidate(
    feature_set: FeatureSet,
    model_name: str,
    model_tag: str,
    class_weight: str | None,
    model: ModelWrapper,
    scaler: FeatureScaler,
    X_val_scaled: pd.DataFrame,
    y_val: pd.Series,
    val_ohlcv: pd.DataFrame,
) -> Candidate:
    """Evaluate one fitted (feature set, model) combination on VALIDATION only.

    Never touches test -- that's reserved for whichever candidate wins
    across every feature-set/model combination.
    """
    val_pred = model.predict(X_val_scaled)
    val_proba = model.predict_proba(X_val_scaled)

    val_classification = classification_report(y_val, val_pred, val_proba)
    val_trading = trading_report(
        val_pred, val_proba, val_ohlcv, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    )

    return Candidate(
        feature_set_name=feature_set.name,
        feature_columns=feature_set.columns,
        model_name=model_name,
        model_tag=model_tag,
        class_weight=class_weight,
        model=model,
        scaler=scaler,
        model_directory=model_directory_for(model_tag, class_weight, feature_set.name),
        val_macro_f1=val_classification["macro_f1"],
        val_log_loss=val_classification["log_loss"],
        val_classification=val_classification,
        val_trading=val_trading,
    )


def train_and_evaluate_feature_set(
    dataset: pd.DataFrame,
    feature_set: FeatureSet,
) -> tuple[TrainingResult, list[Candidate]]:
    """Train LogReg and RF on one feature-set configuration, VALIDATION only.

    Both models share one chronological split and one scaler, fit
    specifically on this feature set's columns -- the same pattern the
    single-feature-set version of this script used, just repeated once
    per feature set instead of once overall.

    Returns the LogReg TrainingResult (its train/val/test partitions are
    identical to what a RandomForest-only run would have produced, since
    chronological_split never depends on which columns are selected) so
    the caller can look up the winner's exact split later without
    recomputing it.
    """
    logreg_result = train_model(
        LogisticRegressionModel(class_weight=LOGREG_CLASS_WEIGHT, random_state=RANDOM_STATE),
        dataset, horizon=HORIZON, train_pct=TRAIN_PCT, val_pct=VAL_PCT,
        feature_columns=feature_set.columns,
    )

    X_train, y_train = separate_features_and_target(
        logreg_result.train, feature_columns=feature_set.columns,
    )
    X_train_scaled = apply_scaler(logreg_result.scaler, X_train)

    rf_model = RandomForestModel(class_weight=RF_CLASS_WEIGHT, random_state=RANDOM_STATE)
    rf_model.fit(X_train_scaled, y_train)

    X_val, y_val = separate_features_and_target(logreg_result.val, feature_columns=feature_set.columns)
    X_val_scaled = apply_scaler(logreg_result.scaler, X_val)

    candidates = [
        evaluate_candidate(
            feature_set, "LogisticRegression", "logreg", LOGREG_CLASS_WEIGHT,
            logreg_result.model, logreg_result.scaler,
            X_val_scaled, y_val, logreg_result.val,
        ),
        evaluate_candidate(
            feature_set, "RandomForest", "rf", RF_CLASS_WEIGHT,
            rf_model, logreg_result.scaler,
            X_val_scaled, y_val, logreg_result.val,
        ),
    ]

    return logreg_result, candidates


def print_candidate_summary_table(candidates: list[Candidate]) -> None:
    """Print one compact row per (feature set, model) candidate.

    With 4 feature sets x 2 models = 8 candidates, printing the full
    classification + trading report for every one (as the earlier
    2-candidate version of this script did) would be a wall of text.
    Full detail is printed only for the overall winner, below.
    """
    header = (
        f"{'Feature set':<22} {'Model':<20} {'Class weight':<14} "
        f"{'Val macro-F1':>14} {'Val log loss':>14}"
    )
    print("\n=== Candidate comparison (VALIDATION set only -- test untouched) ===")
    print(header)
    print("-" * len(header))
    for candidate in candidates:
        class_weight_display = candidate.class_weight if candidate.class_weight is not None else "none"
        print(
            f"{candidate.feature_set_name:<22} {candidate.model_name:<20} "
            f"{class_weight_display:<14} {candidate.val_macro_f1:>14.4f} "
            f"{candidate.val_log_loss:>14.4f}",
        )


def main(provider=None) -> None:
    provider = provider or TwelveDataProvider()
    repository = CSVRepository()

    raw_ohlcv = load_or_fetch_raw_ohlcv(repository, provider)

    print("Building feature dataset...")
    dataset = build_feature_dataset(raw_ohlcv, horizon=HORIZON, threshold=THRESHOLD)
    print(f"Feature dataset: {len(dataset)} rows after warm-up/horizon trimming.")

    # --- Candidate comparison: 4 feature sets x 2 models = 8 candidates,
    # every one evaluated on VALIDATION only. None has touched test yet. ---
    candidates: list[Candidate] = []
    training_results: dict[str, TrainingResult] = {}

    for feature_set in FEATURE_SETS:
        print(f"\nTraining feature set '{feature_set.name}' ({len(feature_set.columns)} features)...")
        result, feature_set_candidates = train_and_evaluate_feature_set(dataset, feature_set)
        training_results[feature_set.name] = result
        candidates.extend(feature_set_candidates)

    print_candidate_summary_table(candidates)

    winner = max(candidates, key=lambda candidate: candidate.val_macro_f1)
    winner_result = training_results[winner.feature_set_name]

    print(
        f"\nWinner: {winner.model_name} on feature set '{winner.feature_set_name}' "
        f"(val macro-F1={winner.val_macro_f1:.4f})",
    )

    print(f"\n=== {winner.model_name} ({winner.feature_set_name}) "
          f"-- classification report (VALIDATION set) ===")
    print(winner.val_classification)
    print(f"\n=== {winner.model_name} ({winner.feature_set_name}) "
          f"-- trading report (VALIDATION set) ===")
    print(winner.val_trading)

    if winner.model_name == "RandomForest":
        print("\n=== Random Forest -- feature importances ===")
        print(winner.model.get_feature_importances())

    # --- Winner only: ONE final, one-time test-set check, using the
    # exact split the winner was trained and validated on. No other
    # candidate is ever evaluated on test. ---
    X_train, y_train = separate_features_and_target(
        winner_result.train, feature_columns=winner.feature_columns,
    )
    X_test, y_test = separate_features_and_target(
        winner_result.test, feature_columns=winner.feature_columns,
    )
    X_test_scaled = apply_scaler(winner.scaler, X_test)

    y_pred = winner.model.predict(X_test_scaled)
    y_proba = winner.model.predict_proba(X_test_scaled)

    print(f"\n=== {winner.model_name} ({winner.feature_set_name}) "
          f"-- classification report (TEST set, final check only) ===")
    test_classification = classification_report(y_test, y_pred, y_proba)
    print(test_classification)

    print(f"\n=== {winner.model_name} ({winner.feature_set_name}) "
          f"-- trading report (TEST set, final check only) ===")
    test_trading = trading_report(
        y_pred, y_proba, winner_result.test, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    print(test_trading)

    # Baselines are a fixed reference floor, not candidates being chosen
    # between, so evaluating them on test carries none of the
    # repeated-peeking risk that touching test for all 8 ML candidates would.
    majority_baseline = MajorityClassBaseline()
    majority_baseline.fit(X_train, y_train)
    majority_pred = majority_baseline.predict(X_test)
    majority_proba = majority_baseline.predict_proba(X_test)
    print("\n=== Majority-class baseline -- classification report (test set) ===")
    print(classification_report(y_test, majority_pred, majority_proba))

    rule_based_baseline = RuleBasedBaseline()
    rule_pred = rule_based_baseline.predict(winner_result.test)
    rule_proba = rule_based_baseline.predict_proba(winner_result.test)
    print("\n=== Rule-based baseline -- classification report (test set) ===")
    print(classification_report(y_test, rule_pred, rule_proba))

    evaluation_metrics = {"classification": test_classification, "trading": test_trading}
    if winner.model_name == "RandomForest":
        evaluation_metrics["feature_importances"] = winner.model.get_feature_importances()

    print(f"\nPersisting winning model ({winner.model_name}, '{winner.feature_set_name}')...")
    metadata = ModelMetadata(
        model_type=f"{winner.model_name}Model",
        hyperparameters=winner.model.get_hyperparameters(),
        feature_columns=winner.feature_columns,
        random_state=RANDOM_STATE,
        train_pct=TRAIN_PCT,
        val_pct=VAL_PCT,
        horizon=HORIZON,
        threshold=THRESHOLD,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        train_start=str(winner_result.train.index.min()),
        train_end=str(winner_result.train.index.max()),
        val_start=str(winner_result.val.index.min()),
        val_end=str(winner_result.val.index.max()),
        test_start=str(winner_result.test.index.min()),
        test_end=str(winner_result.test.index.max()),
        class_distribution=build_class_distribution(y_train),
        evaluation_metrics=evaluation_metrics,
        **current_library_versions(),
    )
    save_model(winner.model_directory, winner.model, winner.scaler, metadata)
    print(f"Model saved to {winner.model_directory}")

    log_experiment(
        EXPERIMENT_LOG_PATH, metadata, winner.model_directory,
        notes=(
            f"Selected by validation macro-F1 over {len(candidates)} candidates "
            f"(4 feature sets x 2 models): {winner.model_name} on "
            f"'{winner.feature_set_name}' (macro-F1={winner.val_macro_f1:.4f})"
        ),
    )
    print(f"Experiment logged to {EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()