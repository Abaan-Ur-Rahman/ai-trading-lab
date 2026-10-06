"""Model + scaler + metadata persistence for the ML layer.

Model persistence uses joblib, which handles sklearn/numpy objects well.
The scaler and metadata are stored as separate JSON files -- plain,
diffable, human-readable, and not reliant on joblib's pickle-based format
since they're just numbers and strings.

On load, the persisted feature columns are re-validated against what the
current pipeline can actually build: build_features' columns, plus prefixed
cross-asset columns for each secondary instrument the metadata says the model
was trained with. A model asking for a column the pipeline cannot produce (an
older pipeline version, say) must fail loudly here rather than silently
predicting on misaligned columns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, Field

from features.cross_asset import cross_asset_column_names
from features.feature_builder import BUILD_FEATURES_COLUMNS
from features.scaling import FeatureScaler
from ml.models.base import ModelWrapper

_MODEL_FILENAME = "model.joblib"
_SCALER_FILENAME = "scaler.json"
_METADATA_FILENAME = "metadata.json"


class ModelMetadata(BaseModel):
    """Everything needed to understand, audit, and safely reload a trained model."""

    model_type: str
    hyperparameters: dict
    feature_columns: list[str]
    random_state: int

    train_pct: float
    val_pct: float
    horizon: int
    threshold: float

    symbol: str
    timeframe: str

    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

    class_distribution: dict[str, int]

    python_version: str
    pandas_version: str
    numpy_version: str
    sklearn_version: str

    evaluation_metrics: dict = Field(default_factory=dict)

    # Secondary instruments the model's cross-asset features need, as
    # {name: provider symbol}, e.g. {"EURUSD": "EUR/USD"}. The name is the
    # column prefix (lower-cased: eurusd_rolling_correlation). Empty for a
    # single-symbol model, which keeps older metadata.json files loadable.
    secondary_symbols: dict[str, str] = Field(default_factory=dict)
    cross_asset_corr_window: int = 20


def current_library_versions() -> dict[str, str]:
    """Return the currently installed python/pandas/numpy/scikit-learn versions.

    Convenience for callers building a ModelMetadata instance, so every
    training script doesn't need to import sys/pandas/numpy/sklearn itself
    just to fill in these four fields.
    """
    return {
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }


def save_model(
    directory: Path,
    model: ModelWrapper,
    scaler: FeatureScaler,
    metadata: ModelMetadata,
) -> None:
    """Persist a fitted model, its scaler, and its metadata to `directory`.

    Creates `directory` (and parents) if it doesn't exist. Writes three
    files: model.joblib, scaler.json, metadata.json. Overwrites any
    existing files of the same name.
    """
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, directory / _MODEL_FILENAME)

    scaler_payload = {"mean": scaler.mean.to_dict(), "std": scaler.std.to_dict()}
    (directory / _SCALER_FILENAME).write_text(json.dumps(scaler_payload, indent=2))

    (directory / _METADATA_FILENAME).write_text(metadata.model_dump_json(indent=2))


def load_model(directory: Path) -> tuple[ModelWrapper, FeatureScaler, ModelMetadata]:
    """Load a model, its scaler, and its metadata from `directory`.

    Raises:
        ValueError: If the persisted feature columns include anything the
            current pipeline cannot build (given the secondary instruments
            recorded in the metadata), or repeat a column, since that means
            the model was trained against a different, incompatible feature
            pipeline and must not silently be used for predictions.
        FileNotFoundError: If `directory` is missing any of the three
            expected files.
    """
    metadata = ModelMetadata.model_validate_json((directory / _METADATA_FILENAME).read_text())

    buildable = set(BUILD_FEATURES_COLUMNS) | set(cross_asset_column_names(list(metadata.secondary_symbols)))
    unknown = [column for column in metadata.feature_columns if column not in buildable]
    if unknown or len(set(metadata.feature_columns)) != len(metadata.feature_columns):
        raise ValueError(
            f"Model was trained with feature columns {metadata.feature_columns}, "
            f"which the current pipeline cannot reproduce (unknown: {unknown}; "
            f"secondary instruments recorded: {list(metadata.secondary_symbols)}). "
            "Refusing to load a model whose features do not match the current pipeline.",
        )

    model = joblib.load(directory / _MODEL_FILENAME)

    scaler_payload = json.loads((directory / _SCALER_FILENAME).read_text())
    scaler = FeatureScaler(
        mean=pd.Series(scaler_payload["mean"]),
        std=pd.Series(scaler_payload["std"]),
    )

    return model, scaler, metadata