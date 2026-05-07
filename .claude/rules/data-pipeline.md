# Data Pipeline Conventions

> Conventions for working with Parquet files and DuckDB in unusual-whales-mcp. Extends [../../../.claude/rules/common/patterns.md](../../../.claude/rules/common/patterns.md).

## Overview

This project's data stack:
- **Storage**: Parquet files (5-10x smaller than CSV, much faster)
- **Query engine**: DuckDB (in-process columnar analytics)
- **Data interface**: pandas DataFrames (analysis, formatting)

The data flow: CSV → Parquet → DuckDB → DataFrame → JSON result

## Data Loader API

All data access goes through `shared.data_loader`:

```python
from shared.data_loader import load_data, list_available_dates, get_latest_file, get_file_by_date
```

### 1. Load the Latest File

```python
df = load_data("options")
```

Finds the most recent `bot-eod-report-*.parquet` in `~/Documents/Stocks/All Options/`.

### 2. Load a Specific Date

```python
df = load_data("options", date="2025-05-07")
```

Finds the file matching that date in the folder. If the exact date isn't available, raises `FileNotFoundError` with available dates.

### 3. Column Pushdown (Performance!)

Load **only** the columns you need:

```python
df = load_data(
    "options",
    date=date,
    usecols=["ticker", "call_volume", "put_volume", "premium"]
)
```

DuckDB pushes the column filter down to the Parquet reader. This is **much faster** than loading all columns and selecting later.

**Always** use `usecols` in production code. Exception: exploratory analysis or debugging.

### 4. Row Limits

```python
df = load_data("options", nrows=1000)
```

Equivalent to SQL's `LIMIT 1000`. Useful for testing or limiting result sets.

### 5. Available Data Types

Map in `shared.data_loader.FOLDER_MAP`:

| Key | Folder | File Pattern |
|-----|--------|--------------|
| `"options"` | All Options | `bot-eod-report-*.parquet` |
| `"darkpool"` | Dark pool | `dp-eod-report-*.parquet` |
| `"hotchains"` | Hot Option Chains | `hot-chains-*.parquet` |
| `"screener"` | Stock Screener | `stock-screener-*.parquet` |
| `"oi"` | OI changes | `chain-oi-changes-*.parquet` |

### 6. List Available Dates

```python
from shared.data_loader import list_available_dates

dates = list_available_dates("options")
# Returns: ["2025-05-07", "2025-05-06", "2025-05-05", ...]
```

Use this for validation or showing users what data is available.

## DuckDB Queries

For complex analyses, use DuckDB directly:

```python
import duckdb
from shared.data_loader import get_latest_file

path = get_latest_file("darkpool")

result = duckdb.query(f"""
    SELECT 
        ticker,
        SUM(volume) as total_volume,
        SUM(premium) as total_premium,
        COUNT(*) as trade_count
    FROM read_parquet('{path}')
    WHERE premium > 1000
    GROUP BY ticker
    ORDER BY total_premium DESC
    LIMIT 20
""").df()
```

### Key Patterns

**Column names with spaces/special chars**: Quote them:
```python
df = duckdb.query(f'SELECT "Ticker Name", "Prem" FROM read_parquet(...)')
```

**Dates**: Filter efficiently in DuckDB, not in pandas:
```python
# Good: Filter in query
result = duckdb.query(f"""
    SELECT * FROM read_parquet('{path}')
    WHERE trade_date >= '2025-05-01' AND trade_date < '2025-05-08'
""").df()

# Bad: Load all, filter in pandas
df = pd.read_parquet(path)
df = df[df['trade_date'] >= '2025-05-01']
```

**Multiple files**: Union them:
```python
import glob
files = glob.glob(str(STOCKS_DIR / "All Options" / "*.parquet"))
result = duckdb.query(f"""
    SELECT * FROM read_parquet({files})
    WHERE trade_date >= '2025-05-01'
""").df()
```

## Performance Best Practices

### 1. Always Use Column Pushdown

```python
# ✅ Good
df = load_data("options", usecols=["ticker", "volume"])

# ❌ Avoid
df = load_data("options")  # Loads all ~50 columns
df = df[["ticker", "volume"]]
```

**Why**: Parquet is columnar. Unneeded columns are never read from disk.

### 2. Filter in DuckDB, Not Pandas

```python
# ✅ Good
result = duckdb.query(f"""
    SELECT * FROM read_parquet('{path}')
    WHERE volume > 10000 AND premium > 1000
""").df()

# ❌ Avoid
df = pd.read_parquet(path)
df = df[(df['volume'] > 10000) & (df['premium'] > 1000)]
```

**Why**: DuckDB processes in parallel; pandas is single-threaded. 10x speedup for large files.

### 3. Limit Result Rows Early

```python
# ✅ Good (limit in query)
result = duckdb.query(f"""
    SELECT * FROM read_parquet('{path}')
    ORDER BY premium DESC
    LIMIT 100
""").df()

# ❌ Avoid (sort everything, then limit)
df = pd.read_parquet(path)
result = df.nlargest(100, 'premium')
```

**Why**: DuckDB's limit is a top-k algorithm, not a full sort.

### 4. Aggregate in DuckDB

```python
# ✅ Good
result = duckdb.query(f"""
    SELECT ticker, SUM(volume) as total_volume
    FROM read_parquet('{path}')
    GROUP BY ticker
""").df()

# ❌ Avoid
df = pd.read_parquet(path)
result = df.groupby('ticker')['volume'].sum()
```

**Why**: DuckDB's GROUP BY is vectorized; much faster on large data.

## DataFrame Conventions

After loading with `load_data()` or DuckDB, follow pandas conventions:

### 1. Column Names

Use lowercase with underscores:
```python
df.columns = [c.lower().replace(" ", "_") for c in df.columns]
```

This ensures consistency across servers.

### 2. Index

Reset the index before returning:
```python
df = df.reset_index(drop=True)
```

**Why**: When serialized to JSON, the index can cause confusion.

### 3. Data Types

Ensure correct types before operations:
```python
df["premium"] = pd.to_numeric(df["premium"], errors='coerce')
df["trade_date"] = pd.to_datetime(df["trade_date"])
```

**Why**: Parquet preserves types, but CSV imports may have guessed wrong.

### 4. Null Handling

Check for NaN values that might have slipped through:
```python
if df.isnull().any().any():
    df = df.dropna()  # or fillna(0), depending on context
```

## Result Formatting

Use `shared.server_utils` for consistent formatting:

```python
from shared.server_utils import text_result, df_to_result

# For DataFrames
return df_to_result(df, max_rows=50)

# For structured data
return text_result({
    "count": len(df),
    "top_ticker": df.iloc[0]["ticker"],
    "data": df.to_dict("records")
})
```

### JSON Serialization

DuckDB + pandas use `.default_handler=str` to handle edge cases:

```python
df.to_json(orient="records", indent=2, default_handler=str)
```

**Why**: Dates, decimals, numpy types may not serialize natively.

## Data Validation

Before analysis, validate the data:

```python
def validate_dataframe(df, required_cols):
    """Check that a DataFrame has all required columns."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")
    return df
```

Use in tools:
```python
@server.call_tool(name="my_tool")
async def call_my_tool(...) -> list:
    df = load_data("options", usecols=["ticker", "volume"])
    df = validate_dataframe(df, ["ticker", "volume"])
    ...
```

## CSV to Parquet Conversion

The `convert.py` script handles CSV → Parquet:

```bash
python convert.py           # Convert all new CSVs
python convert.py --revert  # Roll back (Parquet → CSV)
```

**Workflow**:
1. Download CSVs from Unusual Whales
2. Drop them into `~/Documents/Stocks/<subfolder>/`
3. Run `python convert.py` (idempotent; skips already-converted)
4. Tools automatically pick up the new Parquet files

**Don't**:
- Manually edit CSV files in the folder (convert.py will reprocess them)
- Read CSVs directly in tools (read Parquet instead via `load_data()`)

## Debugging

### Inspect a Parquet File

```python
import duckdb

path = "~/Documents/Stocks/All Options/bot-eod-report-2025-05-07.parquet"
schema = duckdb.query(f"SELECT * FROM read_parquet('{path}') LIMIT 0").description
print(schema)  # Shows column names and types

# Or peek at data
df = duckdb.query(f"SELECT * FROM read_parquet('{path}') LIMIT 5").df()
print(df.head())
```

### Check Available Dates

```python
from shared.data_loader import list_available_dates

for data_type in ["options", "darkpool", "hotchains", "screener", "oi"]:
    dates = list_available_dates(data_type)
    print(f"{data_type}: {dates[:3]}")
```

### Profile a Query

```python
import time

start = time.time()
df = load_data("options", usecols=["ticker", "volume"])
elapsed = time.time() - start
print(f"Loaded {len(df)} rows in {elapsed:.2f}s")
```

## Testing Data Access

Use pytest with mock Parquet files or real data:

```python
# tests/conftest.py
import pytest
from pathlib import Path
import pandas as pd

@pytest.fixture
def sample_parquet(tmp_path):
    """Create a small test Parquet file."""
    df = pd.DataFrame({
        "ticker": ["AAPL", "TSLA", "MSFT"],
        "volume": [1000, 2000, 3000],
        "premium": [10000, 20000, 30000],
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return path
```

Then use in tests:
```python
def test_load_data(sample_parquet, monkeypatch):
    # Monkeypatch STOCKS_DIR to use test data
    from shared.data_loader import STOCKS_DIR, FOLDER_MAP
    monkeypatch.setattr("shared.data_loader.STOCKS_DIR", sample_parquet.parent)
    
    df = load_data("options")
    assert len(df) == 3
```

---

**Related**: See [mcp-server-patterns.md](mcp-server-patterns.md) for how to use `load_data()` in tools and [financial-security.md](financial-security.md) for sensitive data handling.
