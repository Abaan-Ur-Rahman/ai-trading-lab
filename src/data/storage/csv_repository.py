"""Repository for saving and loading market data as CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class CSVRepository:
    """Handles saving and loading pandas DataFrames as CSV files."""

    def save(
            self,
            dataframe: pd.DataFrame,
            path: Path,
    ) -> None:
        """Save a DataFrame to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)

    def load(
        self,
        path: Path,
    ) -> pd.DataFrame:
        """Load a DataFrame from a CSV file.

        Raises:
            FileNotFoundError: If no file exists at `path`.
        """
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        return pd.read_csv(path)

    def exists(
        self,
        path: Path,
    ) -> bool:
        """Return True if the CSV file exists."""
        return path.exists()