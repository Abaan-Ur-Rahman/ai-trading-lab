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
    ...


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """Loading a missing file should raise FileNotFoundError."""
    ...


def test_exists_returns_true_when_file_exists(tmp_path: Path) -> None:
    """exists() should detect existing files."""
    ...