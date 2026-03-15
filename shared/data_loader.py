"""Shared data loader for Unusual Whales CSV exports in ~/Documents/Stocks/."""

from pathlib import Path

import pandas as pd

STOCKS_DIR = Path.home() / "Documents" / "Stocks"

FOLDER_MAP = {
    "options": "All Options",
    "darkpool": "Dark pool",
    "hotchains": "Hot Option Chains",
    "screener": "Stock Screener",
    "oi": "OI changes",
}


def get_data_dir(data_type: str) -> Path:
    """Get the directory for a specific data type."""
    folder = FOLDER_MAP.get(data_type)
    if not folder:
        raise ValueError(f"Unknown data type: {data_type}. Valid: {list(FOLDER_MAP.keys())}")
    return STOCKS_DIR / folder


def list_available_dates(data_type: str) -> list[str]:
    """List all available CSV dates for a data type, newest first."""
    data_dir = get_data_dir(data_type)
    if not data_dir.exists():
        return []
    files = sorted(data_dir.glob("*.csv"), reverse=True)
    dates = []
    for f in files:
        parts = f.stem.split("-")
        # Extract date portion (last 3 parts: YYYY-MM-DD)
        if len(parts) >= 3:
            date_str = "-".join(parts[-3:])
            dates.append(date_str)
    return dates


def get_latest_file(data_type: str) -> Path:
    """Get the most recent CSV file for a data type."""
    data_dir = get_data_dir(data_type)
    files = sorted(data_dir.glob("*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files[0]


def get_file_by_date(data_type: str, date: str) -> Path:
    """Get a specific CSV file by date (YYYY-MM-DD)."""
    data_dir = get_data_dir(data_type)
    matches = [f for f in data_dir.glob("*.csv") if date in f.name]
    if not matches:
        available = list_available_dates(data_type)
        raise FileNotFoundError(
            f"No file found for date {date}. Available: {available}"
        )
    return matches[0]


def load_data(data_type: str, date: str | None = None, **read_kwargs) -> pd.DataFrame:
    """Load a CSV file for a data type, optionally by date.

    If no date given, loads the most recent file.
    Extra kwargs are passed to pd.read_csv (e.g., usecols, nrows).
    """
    path = get_file_by_date(data_type, date) if date else get_latest_file(data_type)
    return pd.read_csv(path, **read_kwargs)
