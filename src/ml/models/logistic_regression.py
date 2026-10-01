"""Logistic Regression model wrapper with a fixed probability class order."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.dataset import CLASS_LABELS, CLASS_NAMES
from ml.models.base import ModelWrapper

_LABEL_TO_NAME = dict(zip(CLASS_LABELS, CLASS_NAMES))


class LogisticRegressionModel(ModelWrapper):
    """Multiclass Logistic Regression wrapper for SELL/HOLD/BUY classification.

    predict_proba() always returns columns in [SELL, HOLD, BUY] order
    (CLASS_NAMES), regardless of the order scikit-learn happens to discover
    classes in during fit. In practice sklearn sorts `classes_` numerically,
    so for our labels [-1, 0, 1] the raw output already matches -- but that
    is an incidental implementation detail, not a documented contract, so
    we reindex explicitly using `model.classes_` rather than assuming
    position. This also protects us if the label scheme ever changes.
    """

    def __init__(
        self,
        class_weight: str | None = None,
        random_state: int = 42,
    ) -> None:
        self._class_weight = class_weight
        self._random_state = random_state
        self._model = LogisticRegression(
            class_weight=class_weight,
            random_state=random_state,
            max_iter=1000,
        )
        self._is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the underlying LogisticRegression on training data."""
        self._model.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted class label (-1, 0, or 1) for each row.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before calling predict")
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities with columns in [SELL, HOLD, BUY] order.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before calling predict_proba")

        raw_proba = self._model.predict_proba(X)
        raw_columns = [_LABEL_TO_NAME[label] for label in self._model.classes_]
        proba = pd.DataFrame(raw_proba, columns=raw_columns, index=X.index)

        return proba[CLASS_NAMES]