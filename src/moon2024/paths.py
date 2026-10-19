"""Locating the input files, locally (./data) or on Kaggle (/kaggle/input)."""

import os
from pathlib import Path

RAW_AEP = "energydata_complete.csv"
AUTHORS_AEP = "Appliances Energy Prediction.csv"
AUTHORS_URC = "University Residential Complex.csv"

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """MOON_DATA_DIR if set, else ./data in the repository."""
    return Path(os.environ.get("MOON_DATA_DIR", REPO_ROOT / "data"))


def find_file(name: str) -> Path:
    """Find a file by name anywhere under data_dir().

    Kaggle mounts notebook outputs at paths that have changed over time,
    so the file is searched for rather than addressed directly.
    """
    root = data_dir()
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {root}")
    return matches[0]
