"""Model + scaler + metadata persistence for the ML layer.

Model persistence uses joblib, which handles sklearn/numpy objects well.
The scaler and metadata are stored as separate JSON files -- plain,
diffable, human-readable, and not reliant on joblib's pickle-based format
since they're just numbers and strings.

On load, the persisted feature order is re-validated against the live
FEATURE_COLUMNS in ml.dataset: a model trained on a different feature set
(an older pipeline version, say) must fail loudly here rather than
silently predicting on misaligned columns.
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

from features.scaling import FeatureScaler
from ml.dataset import FEATURE_COLUMNS
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
        ValueError: If the persisted feature column order does not match
            the current FEATURE_COLUMNS, since that means the model was
            trained against a different (likely incompatible) feature
            pipeline and must not silently be used for predictions.
        FileNotFoundError: If `directory` is missing any of the three
            expected files.
    """
    metadata = ModelMetadata.model_validate_json((directory / _METADATA_FILENAME).read_text())

    if metadata.feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            f"Model was trained with feature columns {metadata.feature_columns}, "
            f"but the current FEATURE_COLUMNS is {FEATURE_COLUMNS}. Refusing to "
            "load a model whose features do not match the current pipeline.",
        )

    model = joblib.load(directory / _MODEL_FILENAME)

    scaler_payload = json.loads((directory / _SCALER_FILENAME).read_text())
    scaler = FeatureScaler(
        mean=pd.Series(scaler_payload["mean"]),
        std=pd.Series(scaler_payload["std"]),
    )

    return model, scaler, metadata