"""Local JSONL experiment log for the ML layer.

Deliberately minimal per project scope: no MLflow, no database, no remote
tracking service. Every training run appends one JSON line recording its
metadata (hyperparameters, split config, evaluation metrics, etc.) and
where its persisted model lives, so past runs can be compared later
without re-running them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml.persistence import ModelMetadata


def log_experiment(
    log_path: Path,
    metadata: ModelMetadata,
    model_directory: Path,
    notes: str = "",
) -> None:
    """Append one experiment record to the JSONL log at `log_path`.

    Creates `log_path`'s parent directory (and the file itself) if they
    don't exist yet. This is an append-only log -- it never rewrites or
    removes existing entries.

    Args:
        log_path: Path to the .jsonl log file.
        metadata: The ModelMetadata for this training run (already has
            hyperparameters, split config, class distribution, library
            versions, and evaluation metrics).
        model_directory: Where persistence.save_model wrote this run's
            model.joblib/scaler.json/metadata.json, so the log entry can
            point back to the actual persisted model.
        notes: Optional free-text note about this run (e.g. why you tried
            this configuration).
    """
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "model_directory": str(model_directory),
        "notes": notes,
        **metadata.model_dump(),
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")


def load_experiments(log_path: Path) -> pd.DataFrame:
    """Load all experiment records from `log_path` as a DataFrame.

    Returns an empty DataFrame (no rows) if `log_path` does not exist yet
    -- "no experiments logged yet" is a normal state, not an error.
    """
    if not log_path.exists():
        return pd.DataFrame()

    records: list[dict] = []
    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return pd.DataFrame(records)