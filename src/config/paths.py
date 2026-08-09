"""
Centralized project paths for AI Trading Lab.

This module defines important filesystem locations used throughout
the project. By storing paths in one place, the rest of the codebase
avoids hardcoded directory names and remains easier to maintain.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------
# Top-Level Directories
# ---------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESEARCH_DIR = PROJECT_ROOT / "research"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# ---------------------------------------------------------------------
# Data Subdirectories
# ---------------------------------------------------------------------

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"