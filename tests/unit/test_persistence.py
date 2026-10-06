"""Unit tests for save_model / load_model round-tripping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.scaling import apply_scaler, fit_scaler
from ml.dataset import FEATURE_COLUMNS
from ml.models.logistic_regression import LogisticRegressionModel
from ml.persistence import ModelMetadata, current_library_versions, load_model, save_model


@pytest.fixture
def fitted_model_and_scaler():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.Series([-1] * 20 + [0] * 20 + [1] * 20, name="label")

    scaler = fit_scaler(X)
    X_scaled = apply_scaler(scaler, X)

    model = LogisticRegressionModel(class_weight="balanced", random_state=7)
    model.fit(X_scaled, y)

    return model, scaler, X_scaled


def _build_metadata() -> ModelMetadata:
    versions = current_library_versions()
    return ModelMetadata(
        model_type="LogisticRegressionModel",
        hyperparameters={"class_weight": "balanced", "random_state": 7},
        feature_columns=FEATURE_COLUMNS,
        random_state=7,
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
        **versions,
    )


def test_save_creates_expected_files(tmp_path: Path, fitted_model_and_scaler) -> None:
    model, scaler, _ = fitted_model_and_scaler
    metadata = _build_metadata()
    directory = tmp_path / "my_model"

    save_model(directory, model, scaler, metadata)

    assert (directory / "model.joblib").exists()
    assert (directory / "scaler.json").exists()
    assert (directory / "metadata.json").exists()


def test_save_and_load_round_trip_predictions_match(
    tmp_path: Path, fitted_model_and_scaler
) -> None:
    model, scaler, X_scaled = fitted_model_and_scaler
    metadata = _build_metadata()
    directory = tmp_path / "my_model"

    save_model(directory, model, scaler, metadata)
    loaded_model, loaded_scaler, loaded_metadata = load_model(directory)

    np.testing.assert_array_equal(model.predict(X_scaled), loaded_model.predict(X_scaled))
    pd.testing.assert_series_equal(scaler.mean, loaded_scaler.mean)
    pd.testing.assert_series_equal(scaler.std, loaded_scaler.std)
    assert loaded_metadata == metadata


def test_load_rejects_mismatched_feature_columns(
    tmp_path: Path, fitted_model_and_scaler
) -> None:
    model, scaler, _ = fitted_model_and_scaler
    metadata = _build_metadata()
    metadata.feature_columns = ["wrong_feature_a", "wrong_feature_b"]
    directory = tmp_path / "my_model"

    save_model(directory, model, scaler, metadata)

    with pytest.raises(ValueError):
        load_model(directory)


def _fit_on(columns: list[str]):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(60, len(columns))), columns=columns)
    y = pd.Series([-1] * 20 + [0] * 20 + [1] * 20, name="label")
    scaler = fit_scaler(X)
    model = LogisticRegressionModel(random_state=7)
    model.fit(apply_scaler(scaler, X), y)
    return model, scaler


CROSS_ASSET_MODEL_COLUMNS = FEATURE_COLUMNS + [
    "eurusd_secondary_log_return",
    "eurusd_ratio_log_return",
    "eurusd_rolling_correlation",
]


def test_load_accepts_cross_asset_columns_when_secondaries_recorded(tmp_path: Path) -> None:
    model, scaler = _fit_on(CROSS_ASSET_MODEL_COLUMNS)
    metadata = _build_metadata()
    metadata.feature_columns = CROSS_ASSET_MODEL_COLUMNS
    metadata.secondary_symbols = {"EURUSD": "EUR/USD"}
    directory = tmp_path / "cross_asset_model"

    save_model(directory, model, scaler, metadata)
    _, _, loaded = load_model(directory)

    assert loaded.feature_columns == CROSS_ASSET_MODEL_COLUMNS
    assert loaded.secondary_symbols == {"EURUSD": "EUR/USD"}


def test_load_rejects_cross_asset_columns_without_recorded_secondaries(tmp_path: Path) -> None:
    model, scaler = _fit_on(CROSS_ASSET_MODEL_COLUMNS)
    metadata = _build_metadata()
    metadata.feature_columns = CROSS_ASSET_MODEL_COLUMNS
    directory = tmp_path / "cross_asset_model"

    save_model(directory, model, scaler, metadata)

    with pytest.raises(ValueError):
        load_model(directory)


def test_load_rejects_duplicate_feature_columns(tmp_path: Path, fitted_model_and_scaler) -> None:
    model, scaler, _ = fitted_model_and_scaler
    metadata = _build_metadata()
    metadata.feature_columns = FEATURE_COLUMNS + ["rsi"]
    directory = tmp_path / "my_model"

    save_model(directory, model, scaler, metadata)

    with pytest.raises(ValueError):
        load_model(directory)


def test_metadata_without_secondary_fields_still_loads() -> None:
    """metadata.json files written before secondary_symbols existed must stay readable."""
    payload = _build_metadata().model_dump()
    del payload["secondary_symbols"], payload["cross_asset_corr_window"]

    loaded = ModelMetadata.model_validate(payload)

    assert loaded.secondary_symbols == {}
    assert loaded.cross_asset_corr_window == 20


def test_metadata_round_trips_nested_evaluation_metrics(
    tmp_path: Path, fitted_model_and_scaler
) -> None:
    model, scaler, _ = fitted_model_and_scaler
    metadata = _build_metadata()
    directory = tmp_path / "my_model"

    save_model(directory, model, scaler, metadata)
    _, _, loaded_metadata = load_model(directory)

    assert loaded_metadata.evaluation_metrics == metadata.evaluation_metrics


def test_current_library_versions_returns_all_four_keys() -> None:
    versions = current_library_versions()

    assert set(versions.keys()) == {
        "python_version",
        "pandas_version",
        "numpy_version",
        "sklearn_version",
    }