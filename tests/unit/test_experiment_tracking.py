"""Unit tests for log_experiment / load_experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.dataset import FEATURE_COLUMNS
from ml.experiment_tracking import load_experiments, log_experiment
from ml.persistence import ModelMetadata, current_library_versions


def _build_metadata(**overrides) -> ModelMetadata:
    defaults = dict(
        model_type="LogisticRegressionModel",
        hyperparameters={"class_weight": None, "random_state": 42},
        feature_columns=FEATURE_COLUMNS,
        random_state=42,
        train_pct=0.70,
        val_pct=0.15,
        horizon=5,
        threshold=0.005,
        symbol="EURUSD",
        timeframe="1h",
        train_start="2020-01-01",
        train_end="2020-06-01",
        val_start="2020-06-02",
        val_end="2020-08-01",
        test_start="2020-08-02",
        test_end="2020-10-01",
        class_distribution={"SELL": 20, "HOLD": 20, "BUY": 20},
        evaluation_metrics={"accuracy": 0.8, "per_class": {"BUY": {"precision": 0.7}}},
        **current_library_versions(),
    )
    defaults.update(overrides)
    return ModelMetadata(**defaults)


def test_log_experiment_creates_file_and_parent_dirs(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "experiments.jsonl"
    metadata = _build_metadata()

    log_experiment(log_path, metadata, model_directory=tmp_path / "models" / "run1")

    assert log_path.exists()


def test_log_experiment_appends_without_overwriting(tmp_path: Path) -> None:
    log_path = tmp_path / "experiments.jsonl"

    log_experiment(log_path, _build_metadata(symbol="EURUSD"), tmp_path / "run1")
    log_experiment(log_path, _build_metadata(symbol="GBPUSD"), tmp_path / "run2")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_load_experiments_returns_empty_dataframe_when_file_missing(tmp_path: Path) -> None:
    result = load_experiments(tmp_path / "does_not_exist.jsonl")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_load_experiments_round_trips_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "experiments.jsonl"
    metadata = _build_metadata(symbol="EURUSD")
    model_directory = tmp_path / "models" / "run1"

    log_experiment(log_path, metadata, model_directory, notes="first MVP run")
    result = load_experiments(log_path)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["symbol"] == "EURUSD"
    assert row["model_directory"] == str(model_directory)
    assert row["notes"] == "first MVP run"
    assert row["evaluation_metrics"] == {"accuracy": 0.8, "per_class": {"BUY": {"precision": 0.7}}}
    assert "logged_at" in row


def test_load_experiments_handles_multiple_runs(tmp_path: Path) -> None:
    log_path = tmp_path / "experiments.jsonl"

    log_experiment(log_path, _build_metadata(symbol="EURUSD"), tmp_path / "run1")
    log_experiment(log_path, _build_metadata(symbol="GBPUSD"), tmp_path / "run2")

    result = load_experiments(log_path)

    assert len(result) == 2
    assert set(result["symbol"]) == {"EURUSD", "GBPUSD"}