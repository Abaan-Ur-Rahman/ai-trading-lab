"""Random Forest model wrapper with a fixed probability class order."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.dataset import CLASS_LABELS, CLASS_NAMES
from ml.models.base import ModelWrapper

_LABEL_TO_NAME = dict(zip(CLASS_LABELS, CLASS_NAMES))


class RandomForestModel(ModelWrapper):
    """Random Forest wrapper for SELL/HOLD/BUY classification.

    Hyperparameter defaults are deliberately conservative (max_depth=5,
    min_samples_leaf=20) rather than sklearn's own defaults
    (max_depth=None, min_samples_leaf=1). An unconstrained forest grows
    until every leaf is pure, which on noisy financial return data means
    memorizing noise rather than learning signal -- these defaults are a
    starting regularization point, not a tuned result.

    predict_proba() reindexes to [SELL, HOLD, BUY] column order for the
    same reason as LogisticRegressionModel: sklearn sorts classes_
    numerically in practice, but that's not a documented guarantee.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 5,
        min_samples_leaf: int = 20,
        class_weight: str | None = None,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._class_weight = class_weight
        self._random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._is_fitted = False
        self._feature_names: list[str] | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the underlying RandomForestClassifier on training data."""
        self._model.fit(X_train, y_train)
        self._feature_names = list(X_train.columns)
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

    def get_hyperparameters(self) -> dict:
        """Return this model's hyperparameters as a plain dict."""
        return {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "min_samples_leaf": self._min_samples_leaf,
            "class_weight": self._class_weight,
            "random_state": self._random_state,
        }

    def get_feature_importances(self) -> dict[str, float]:
        """Return a mapping of feature name to impurity-based importance.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fit before calling get_feature_importances")

        return dict(zip(self._feature_names, self._model.feature_importances_.tolist()))