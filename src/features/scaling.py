"""Fit and apply feature scaling for the ML pipeline.

Scaling must be fit on the training partition only, after the chronological
train/test split -- fitting on the full dataset (including test data) would
leak information about the future into the scaling parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureScaler:
    """Holds the mean and standard deviation fitted from training data."""

    mean: pd.Series
    std: pd.Series


def fit_scaler(training_features: pd.DataFrame) -> FeatureScaler:
    """Fit a scaler using only the given (training) data.

    Raises ValueError if any column has zero variance, since dividing by
    a zero standard deviation is undefined.
    """
    mean = training_features.mean()
    std = training_features.std()

    zero_variance_columns = std[std == 0].index.tolist()
    if zero_variance_columns:
        raise ValueError(
            f"Cannot scale columns with zero variance: {zero_variance_columns}",
        )

    return FeatureScaler(mean=mean, std=std)


def apply_scaler(scaler: FeatureScaler, features: pd.DataFrame) -> pd.DataFrame:
    """Apply an already-fitted scaler to features.

    Does not mutate the input dataframe. Does not re-fit -- the same
    mean/std from `scaler` are used regardless of what `features`
    contains, which is what makes it safe to call on validation/test data
    without leaking those rows' own statistics back into the scaling.
    """
    missing = set(scaler.mean.index).difference(features.columns)
    if missing:
        raise ValueError(
            f"features is missing columns the scaler expects: {sorted(missing)}",
        )

    return (features[scaler.mean.index] - scaler.mean) / scaler.std