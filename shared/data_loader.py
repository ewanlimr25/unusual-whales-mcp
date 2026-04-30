"""Shared data loader for Unusual Whales Parquet exports in ~/Documents/Stocks/."""

from pathlib import Path

import duckdb
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
    folder = FOLDER_MAP.get(data_type)
    if not folder:
        raise ValueError(f"Unknown data type: {data_type}. Valid: {list(FOLDER_MAP.keys())}")
    return STOCKS_DIR / folder


def list_available_dates(data_type: str) -> list[str]:
    data_dir = get_data_dir(data_type)
    if not data_dir.exists():
        return []
    dates = []
    for f in sorted(data_dir.glob("*.parquet"), reverse=True):
        parts = f.stem.split("-")
        if len(parts) >= 3:
            dates.append("-".join(parts[-3:]))
    return dates


def get_latest_file(data_type: str) -> Path:
    data_dir = get_data_dir(data_type)
    files = sorted(data_dir.glob("*.parquet"), reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {data_dir}. Run: python convert.py"
        )
    return files[0]


def get_file_by_date(data_type: str, date: str) -> Path:
    data_dir = get_data_dir(data_type)
    matches = [f for f in data_dir.glob("*.parquet") if date in f.name]
    if not matches:
        available = list_available_dates(data_type)
        raise FileNotFoundError(
            f"No Parquet file for {date}. Run: python convert.py. Available: {available}"
        )
    return matches[0]


def load_data(
    data_type: str,
    date: str | None = None,
    usecols: list[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a Parquet file for a data type, optionally filtered by date.

    usecols: column pushdown — only read these columns from Parquet.
    nrows: row limit — equivalent to LIMIT in SQL.
    """
    path = get_file_by_date(data_type, date) if date else get_latest_file(data_type)
    cols = ", ".join(f'"{c}"' for c in usecols) if usecols else "*"
    limit = f" LIMIT {nrows}" if nrows else ""
    return duckdb.query(f"SELECT {cols} FROM read_parquet('{path}'){limit}").df()
