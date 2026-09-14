"""Unit tests for feature scaling."""

from __future__ import annotations

import pandas as pd
import pytest

from features.scaling import FeatureScaler, apply_scaler, fit_scaler


@pytest.fixture
def training_dataframe() -> pd.DataFrame:
    """Return a small training-feature dataframe with known statistics."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def test_fit_scaler_computes_expected_mean_and_std(
    training_dataframe: pd.DataFrame,
) -> None:
    """fit_scaler should compute the exact mean/std of the given data."""
    scaler = fit_scaler(training_dataframe)

    assert scaler.mean["a"] == pytest.approx(3.0)
    assert scaler.mean["b"] == pytest.approx(30.0)
    assert scaler.std["a"] == pytest.approx(1.581139, abs=1e-6)
    assert scaler.std["b"] == pytest.approx(15.811388, abs=1e-6)


def test_fit_scaler_rejects_zero_variance_column() -> None:
    """fit_scaler should raise if any column has zero variance."""
    dataframe = pd.DataFrame({"a": [5.0, 5.0, 5.0]})

    with pytest.raises(ValueError):
        fit_scaler(dataframe)


def test_apply_scaler_scales_training_data_to_mean_zero_std_one(
    training_dataframe: pd.DataFrame,
) -> None:
    """Applying a scaler to the data it was fit on should give mean~0, std~1."""
    scaler = fit_scaler(training_dataframe)

    scaled = apply_scaler(scaler, training_dataframe)

    assert scaled["a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled["a"].std() == pytest.approx(1.0, abs=1e-9)


def test_apply_scaler_uses_training_statistics_not_validation_statistics(
    training_dataframe: pd.DataFrame,
) -> None:
    """Applying a train-fitted scaler to different data must not re-fit.

    This is the leakage-prevention guarantee for this module: validation
    data almost never has mean 0 / std 1 after scaling with training
    stats, and this test proves apply_scaler doesn't quietly compute its
    own statistics from whatever it's given.
    """
    scaler = fit_scaler(training_dataframe)

    validation_dataframe = pd.DataFrame({"a": [6.0, 7.0], "b": [60.0, 70.0]})

    scaled_validation = apply_scaler(scaler, validation_dataframe)

    assert scaled_validation["a"].iloc[0] == pytest.approx(1.897367, abs=1e-6)
    assert scaled_validation["a"].iloc[1] == pytest.approx(2.529822, abs=1e-6)


def test_apply_scaler_requires_expected_columns(
    training_dataframe: pd.DataFrame,
) -> None:
    """apply_scaler should raise if features is missing a column the scaler expects."""
    scaler = fit_scaler(training_dataframe)

    incomplete = pd.DataFrame({"a": [1.0, 2.0]})

    with pytest.raises(ValueError):
        apply_scaler(scaler, incomplete)


def test_apply_scaler_does_not_mutate_input(
    training_dataframe: pd.DataFrame,
) -> None:
    """apply_scaler must not modify the caller's original dataframe."""
    scaler = fit_scaler(training_dataframe)
    original = training_dataframe.copy(deep=True)

    apply_scaler(scaler, training_dataframe)

    pd.testing.assert_frame_equal(training_dataframe, original)