"""Unit tests for CSVRepository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.storage.csv_repository import CSVRepository


def test_save_creates_csv_file(tmp_path: Path) -> None:
    """Saving should create a CSV file."""

    repository = CSVRepository()

    dataframe = pd.DataFrame(
        {
            "close": [100.0, 101.0],
        }
    )

    file_path = tmp_path / "market.csv"

    repository.save(dataframe, file_path)

    assert file_path.exists()


def test_load_returns_dataframe(tmp_path: Path) -> None:
    """Loading should return the saved DataFrame."""
    repository = CSVRepository()
    dataframe = pd.DataFrame({"close": [100.0, 101.0]})
    file_path = tmp_path / "market.csv"
    repository.save(dataframe, file_path)

    loaded = repository.load(file_path)

    pd.testing.assert_frame_equal(loaded, dataframe)


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Loading a missing file should raise FileNotFoundError."""
    repository = CSVRepository()

    with pytest.raises(FileNotFoundError):
        repository.load(tmp_path / "does_not_exist.csv")


def test_exists_returns_true_when_file_exists(tmp_path: Path) -> None:
    """exists() should detect existing files."""
    repository = CSVRepository()
    dataframe = pd.DataFrame({"close": [100.0]})
    file_path = tmp_path / "market.csv"
    repository.save(dataframe, file_path)

    assert repository.exists(file_path) is True


def test_exists_returns_false_when_file_missing(tmp_path: Path) -> None:
    """exists() should return False for a path that doesn't exist."""
    repository = CSVRepository()

    assert repository.exists(tmp_path / "does_not_exist.csv") is False