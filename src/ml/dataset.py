"""Chronological dataset splitting and feature/target separation for the ML layer.

This module is the single boundary between the Feature Engineering layer and
the rest of src/ml/. Every other file in src/ml/ receives already-split X/y
data from here and never imports from src/features/ directly.
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["log_return", "ema_gap_pct", "rsi", "atr_pct", "macd_hist_pct"]

# Fixed class ordering used everywhere downstream (models, evaluation,
# persistence) so probability columns and metrics never depend on the
# incidental order sklearn happens to discover classes in.
CLASS_LABELS = [-1, 0, 1]
CLASS_NAMES = ["SELL", "HOLD", "BUY"]


def chronological_split(
    dataset: pd.DataFrame,
    horizon: int,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a chronologically-ordered dataset into train/val/test partitions.

    Splits are purely positional (no shuffling). A purge gap of exactly
    `horizon` rows is removed at both the train/validation boundary and the
    validation/test boundary, since a row's label was computed from data up
    to `horizon` rows ahead of it -- without the gap, the last training
    label's forward-looking window would extend into the validation period.

    This purging is positional, not index-label based: it assumes `dataset`
    rows are chronologically contiguous with no internal gaps. This holds
    for build_feature_dataset's normal output, which only drops leading rows
    (indicator warm-up) and trailing rows (label horizon), never rows from
    the middle. A dataset with internal gaps (e.g. missing trading days)
    would need index-aware purging, which is out of scope for this MVP.

    Does not mutate the input dataframe.

    Args:
        dataset: Chronologically-ordered feature+label data, e.g. the output
            of build_feature_dataset.
        horizon: The label horizon the dataset was built with (rows to purge
            at each boundary).
        train_pct: Fraction of rows allocated to training, before purging.
        val_pct: Fraction of rows allocated to validation, before purging.
            The remainder (1 - train_pct - val_pct) is allocated to test.

    Returns:
        (train, val, test) DataFrames, each a view-safe slice of `dataset`.

    Raises:
        ValueError: If horizon, train_pct, or val_pct are out of valid
            range, or if any resulting partition is empty after purging.
    """
    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    if not (0 < train_pct < 1):
        raise ValueError("train_pct must be between 0 and 1")

    if not (0 < val_pct < 1):
        raise ValueError("val_pct must be between 0 and 1")

    if train_pct + val_pct >= 1:
        raise ValueError(
            "train_pct + val_pct must be less than 1, leaving room for a test set",
        )

    n_rows = len(dataset)
    train_end = int(n_rows * train_pct)
    val_end = train_end + int(n_rows * val_pct)

    train = dataset.iloc[:train_end]
    val = dataset.iloc[train_end + horizon : val_end]
    test = dataset.iloc[val_end + horizon :]

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            "One or more partitions is empty after applying the purge gap; "
            "provide more data or reduce horizon relative to dataset size",
        )

    return train, val, test


def separate_features_and_target(
    dataset: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature+label dataset into X (features) and y (integer target).

    y is cast to int -- the upstream label column is float64 (it has to be,
    to hold NaN during the feature pipeline's own dropna step), but by the
    time it reaches this function there are no NaNs left, so int gives
    cleaner class labels for sklearn and metric reporting.

    Does not mutate the input dataframe.

    Args:
        dataset: Feature+label data, e.g. the output of
            build_feature_dataset (or a chronological_split partition of
            it).
        feature_columns: Which columns to select as X. Defaults to
            FEATURE_COLUMNS (the module's standard 5-feature set) when
            not provided, so every existing caller is unaffected. Pass
            an explicit list to select a different feature subset --
            e.g. for comparing feature-set configurations against each
            other.

    Raises:
        ValueError: If feature_columns is an empty list, or if any
            requested feature column or 'label' is missing from dataset.
    """
    columns = FEATURE_COLUMNS if feature_columns is None else feature_columns

    if not columns:
        raise ValueError("feature_columns must not be empty")

    missing_features = set(columns).difference(dataset.columns)
    if missing_features:
        raise ValueError(
            f"dataset is missing required feature columns: {sorted(missing_features)}",
        )

    if "label" not in dataset.columns:
        raise ValueError("dataset must contain a 'label' column")

    features = dataset[columns].copy()
    target = dataset["label"].astype(int)
    target.name = "label"

    return features, target


def validate_all_classes_present(y_train: pd.Series) -> None:
    """Ensure the training target contains every class in CLASS_LABELS.

    Fails fast with a clear message if a chronological split happens to
    produce a training partition missing a class (e.g. an all-HOLD window),
    rather than letting sklearn fail with a less specific error later.

    Raises:
        ValueError: If y_train is missing any class in CLASS_LABELS.
    """
    present = set(y_train.unique())
    missing = set(CLASS_LABELS).difference(present)

    if missing:
        raise ValueError(
            f"Training partition is missing required classes: {sorted(missing)}. "
            f"Present classes: {sorted(present)}.",
        )