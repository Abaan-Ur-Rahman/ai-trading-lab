"""Unit tests for label creation."""

from __future__ import annotations

import pandas as pd
import pytest

from features.labeling import create_labels


@pytest.fixture
def sample_close_dataframe() -> pd.DataFrame:
    """Return a small close-price dataframe covering BUY, SELL, and HOLD cases."""
    return pd.DataFrame(
        {"close": [100, 102, 99, 103, 97, 104, 95, 108, 90, 110]},
        dtype=float,
    )


def test_create_labels_returns_expected_values(
    sample_close_dataframe: pd.DataFrame,
) -> None:
    """Labels should match hand-verified BUY/SELL/HOLD outcomes."""
    labels = create_labels(sample_close_dataframe, horizon=2, threshold=0.02)

    assert labels.iloc[0] == 0.0   # HOLD: future_return ~ -0.01
    assert labels.iloc[2] == -1.0  # SELL: future_return ~ -0.0202
    assert labels.iloc[5] == 1.0   # BUY: future_return ~ 0.0385


def test_create_labels_last_horizon_rows_are_nan(
    sample_close_dataframe: pd.DataFrame,
) -> None:
    """The last `horizon` rows have no future data and must be NaN."""
    labels = create_labels(sample_close_dataframe, horizon=2, threshold=0.02)

    assert labels.iloc[-2:].isna().all()
    assert labels.iloc[:-2].notna().all()


def test_create_labels_requires_close_column() -> None:
    """create_labels should require a close column."""
    dataframe = pd.DataFrame({"price": [1, 2, 3]})

    with pytest.raises(ValueError):
        create_labels(dataframe)


def test_create_labels_requires_positive_horizon(
    sample_close_dataframe: pd.DataFrame,
) -> None:
    """horizon must be greater than 0."""
    with pytest.raises(ValueError):
        create_labels(sample_close_dataframe, horizon=0)


def test_create_labels_requires_positive_threshold(
    sample_close_dataframe: pd.DataFrame,
) -> None:
    """threshold must be greater than 0."""
    with pytest.raises(ValueError):
        create_labels(sample_close_dataframe, threshold=0.0)


def test_create_labels_does_not_mutate_input(
    sample_close_dataframe: pd.DataFrame,
) -> None:
    """create_labels must not modify the caller's original dataframe."""
    original = sample_close_dataframe.copy(deep=True)

    create_labels(sample_close_dataframe, horizon=2, threshold=0.02)

    pd.testing.assert_frame_equal(sample_close_dataframe, original)