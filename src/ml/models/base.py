"""Abstract interface for ML model wrappers used in the ML layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ModelWrapper(ABC):
    """Common interface for all ML models in the ML layer.

    Persistence is intentionally NOT part of this interface. A separate
    module (ml/persistence.py, a later step) owns serialization, since it
    needs to bundle the model together with metadata (feature order,
    hyperparameters, library versions, etc.) that has nothing to do with
    the model's own prediction logic. Keeping save/load off this interface
    keeps each concrete model focused only on fit/predict/predict_proba.
    """

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model on the given features and integer target."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted class label (-1, 0, or 1) for each row."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities with columns in [SELL, HOLD, BUY] order."""

    @abstractmethod
    def get_hyperparameters(self) -> dict:
        """Return this model's hyperparameters as a plain dict, for metadata/logging."""