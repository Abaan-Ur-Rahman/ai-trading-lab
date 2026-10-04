"""End-to-end training script: fetch XAUUSD 1h data, train, evaluate, persist.

This is orchestration, not a src/ layer module: every function it calls
(data fetching, feature building, splitting, scaling, tuning, evaluation,
persistence, experiment logging) already has its own unit tests. This
script just wires them together in the right order and is not itself unit
tested, the same way a thin CLI entry point normally isn't -- the logic
worth testing lives one level down, where it's already covered.

Model selection: 4 feature-set configurations (base / plus_momentum /
plus_range / plus_momentum_range) x 2 models (LogisticRegression,
RandomForest) = 8 candidates. Each candidate's hyperparameters are first
selected via chronological cross-validation strictly within the TRAIN
partition (see ml.tuning), then evaluated on VALIDATION only. Whichever
candidate has the higher validation macro-F1 is the winner, decided here
in code -- and only the winner gets the one-time, final TEST-set check and
gets persisted. No other candidate, and no hyperparameter trial, ever
touches validation or test except for the winner's own one-time check.

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
from ml.dataset import (
    CLASS_LABELS,
    CLASS_NAMES,
    FEATURE_COLUMNS,
    chronological_split,
    separate_features_and_target,
)
from ml.evaluation import classification_report, trading_report
from ml.experiment_tracking import log_experiment
from ml.models.base import ModelWrapper
from ml.models.logistic_regression import LogisticRegressionModel
from ml.models.random_forest import RandomForestModel
from ml.persistence import ModelMetadata, current_library_versions, save_model
from ml.tuning import TuningTrial, tune_hyperparameters

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

# Kept intentionally small -- this is model/hyperparameter selection, not
# an aggressive search. See ml.tuning for how these are evaluated
# (chronological CV strictly within the train partition).
LOGREG_PARAM_GRID = {"C": [0.01, 0.1, 1.0, 10.0]}
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 5],
}
TUNING_N_SPLITS = 3

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
    columns cannot be reused for a different feature set. best_hyperparameters
    and tuning_trials come straight from this candidate's TuningResult, so
    the full search history survives for inspection, not just the winner.
    """

    feature_set_name: str
    feature_columns: list[str]
    model_name: str
    model_tag: str
    class_weight: str | None
    model: ModelWrapper
    scaler: FeatureScaler
    model_directory: Path
    best_hyperparameters: dict
    tuning_trials: list[TuningTrial]
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


def tune_and_evaluate_candidate(
    feature_set: FeatureSet,
    model_name: str,
    model_tag: str,
    model_factory,
    param_grid: dict[str, list],
    fixed_params: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> Candidate:
    """Tune one model's hyperparameters on `train`, then evaluate on `val`.

    Hyperparameter selection happens entirely inside tune_hyperparameters,
    strictly within `train` (see ml.tuning for the chronological-CV
    mechanics). The returned model/scaler are already fit on the full
    train partition with the winning hyperparameters -- this function
    only adds the VALIDATION-set evaluation on top. Never touches test.
    """
    tuning_result = tune_hyperparameters(
        model_factory=model_factory,
        param_grid=param_grid,
        fixed_params=fixed_params,
        train=train,
        horizon=HORIZON,
        feature_columns=feature_set.columns,
        n_splits=TUNING_N_SPLITS,
    )

    X_val, y_val = separate_features_and_target(val, feature_columns=feature_set.columns)
    X_val_scaled = apply_scaler(tuning_result.scaler, X_val)

    val_pred = tuning_result.model.predict(X_val_scaled)
    val_proba = tuning_result.model.predict_proba(X_val_scaled)

    val_classification = classification_report(y_val, val_pred, val_proba)
    val_trading = trading_report(
        val_pred, val_proba, val, horizon=HORIZON,
        min_confidence=MIN_CONFIDENCE, transaction_cost_pct=TRANSACTION_COST_PCT,
    )

    class_weight = fixed_params.get("class_weight")

    return Candidate(
        feature_set_name=feature_set.name,
        feature_columns=feature_set.columns,
        model_name=model_name,
        model_tag=model_tag,
        class_weight=class_weight,
        model=tuning_result.model,
        scaler=tuning_result.scaler,
        model_directory=model_directory_for(model_tag, class_weight, feature_set.name),
        best_hyperparameters=tuning_result.best_hyperparameters,
        tuning_trials=tuning_result.trials,
        val_macro_f1=val_classification["macro_f1"],
        val_log_loss=val_classification["log_loss"],
        val_classification=val_classification,
        val_trading=val_trading,
    )


def train_and_evaluate_feature_set(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_set: FeatureSet,
) -> list[Candidate]:
    """Tune and evaluate both models on one feature-set configuration."""
    logreg_candidate = tune_and_evaluate_candidate(
        feature_set, "LogisticRegression", "logreg", LogisticRegressionModel,
        LOGREG_PARAM_GRID, {"class_weight": LOGREG_CLASS_WEIGHT, "random_state": RANDOM_STATE},
        train, val,
    )
    rf_candidate = tune_and_evaluate_candidate(
        feature_set, "RandomForest", "rf", RandomForestModel,
        RF_PARAM_GRID, {"class_weight": RF_CLASS_WEIGHT, "random_state": RANDOM_STATE},
        train, val,
    )
    return [logreg_candidate, rf_candidate]


def print_candidate_summary_table(candidates: list[Candidate]) -> None:
    """Print one compact row per (feature set, model) candidate.

    With 4 feature sets x 2 models = 8 candidates, each with its own
    hyperparameter search, printing full detail for every one would be
    unreadable. Full detail -- including the tuning trial breakdown -- is
    printed only for the overall winner, below.
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

    # One chronological split, shared by every feature set and every model
    # -- the split only depends on dataset length and horizon, never on
    # which columns are selected as features or how a model is tuned.
    train, val, test = chronological_split(dataset, horizon=HORIZON, train_pct=TRAIN_PCT, val_pct=VAL_PCT)

    # --- Candidate comparison: 4 feature sets x 2 models = 8 candidates.
    # Hyperparameters are tuned strictly within `train`; every candidate is
    # then evaluated on VALIDATION only. None has touched test yet. ---
    candidates: list[Candidate] = []
    for feature_set in FEATURE_SETS:
        print(
            f"\nTuning + evaluating feature set '{feature_set.name}' "
            f"({len(feature_set.columns)} features)...",
        )
        candidates.extend(train_and_evaluate_feature_set(train, val, feature_set))

    print_candidate_summary_table(candidates)

    winner = max(candidates, key=lambda candidate: candidate.val_macro_f1)

    print(
        f"\nWinner: {winner.model_name} on feature set '{winner.feature_set_name}' "
        f"(val macro-F1={winner.val_macro_f1:.4f})",
    )
    print(f"Selected hyperparameters: {winner.best_hyperparameters}")

    print(f"\n=== {winner.model_name} ({winner.feature_set_name}) -- hyperparameter tuning trials ===")
    for trial in sorted(winner.tuning_trials, key=lambda t: t.mean_macro_f1, reverse=True):
        fold_scores_display = [f"{score:.4f}" for score in trial.fold_scores]
        print(
            f"  {trial.hyperparameters} -> mean macro-F1={trial.mean_macro_f1:.4f} "
            f"(folds: {fold_scores_display})",
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

    # --- Winner only: ONE final, one-time test-set check. No other
    # candidate, and no hyperparameter trial, is ever evaluated on test. ---
    X_train, y_train = separate_features_and_target(train, feature_columns=winner.feature_columns)
    X_test, y_test = separate_features_and_target(test, feature_columns=winner.feature_columns)
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
        y_pred, y_proba, test, horizon=HORIZON,
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
    rule_pred = rule_based_baseline.predict(test)
    rule_proba = rule_based_baseline.predict_proba(test)
    print("\n=== Rule-based baseline -- classification report (test set) ===")
    print(classification_report(y_test, rule_pred, rule_proba))

    evaluation_metrics = {
        "classification": test_classification,
        "trading": test_trading,
        "hyperparameter_tuning": [
            {
                "hyperparameters": trial.hyperparameters,
                "fold_scores": trial.fold_scores,
                "mean_macro_f1": trial.mean_macro_f1,
            }
            for trial in winner.tuning_trials
        ],
    }
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
        train_start=str(train.index.min()),
        train_end=str(train.index.max()),
        val_start=str(val.index.min()),
        val_end=str(val.index.max()),
        test_start=str(test.index.min()),
        test_end=str(test.index.max()),
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
            f"(4 feature sets x 2 models, each hyperparameter-tuned via "
            f"chronological CV on train only): {winner.model_name} on "
            f"'{winner.feature_set_name}' (macro-F1={winner.val_macro_f1:.4f}), "
            f"hyperparameters={winner.best_hyperparameters}"
        ),
    )
    print(f"Experiment logged to {EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()