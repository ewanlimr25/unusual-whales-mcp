"""Load and manage Unusual Whales data exports."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def list_data_files() -> list[str]:
    """List all CSV/JSON files available in the data directory."""
    files = []
    for ext in ("*.csv", "*.json"):
        files.extend(f.name for f in DATA_DIR.glob(ext))
    return sorted(files)


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")
    if not path.suffix == ".csv":
        raise ValueError(f"Expected a CSV file, got: {filename}")
    return pd.read_csv(path)


def load_json(filename: str) -> pd.DataFrame:
    """Load a JSON file from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")
    if not path.suffix == ".json":
        raise ValueError(f"Expected a JSON file, got: {filename}")
    return pd.read_json(path)


def load_file(filename: str) -> pd.DataFrame:
    """Load a data file (CSV or JSON) from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {filename}. "
            f"Available files: {list_data_files()}"
        )
    if path.suffix == ".csv":
        return load_csv(filename)
    elif path.suffix == ".json":
        return load_json(filename)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
