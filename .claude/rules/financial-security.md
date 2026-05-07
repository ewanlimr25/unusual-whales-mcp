# Financial Data Security

> Security guidelines for handling **sensitive trading data**. This repo contains options flows, dark pool activity, and institutional positions — treat with the same care as user PII. Extends [../../../.claude/rules/common/security.md](../../../.claude/rules/common/security.md).

## Critical: Trading Data is Sensitive

This data reveals:
- Which institutions are accumulating positions
- Unusual options activity before price moves
- Dark pool order sizes and institutional activity

**If exposed**, this data could:
- Leak trading strategies to competitors
- Violate institutional client confidentiality
- Constitute market manipulation if misused

Treat like financial PII. Follow these rules **without exception**.

## Rules

### 1. Never Expose Raw Trades

❌ **Don't** return raw trade-level data with timestamps:
```python
return df_to_result(df)  # Returns every row: timestamp, volume, premium, ...
```

✅ **Do** aggregate and sanitize before returning:
```python
result = df.groupby("ticker").agg({
    "volume": "sum",
    "premium": "sum",
    "trade_count": "count"
}).reset_index()

return df_to_result(result)
```

**Why**: Trade timestamps + size = institutional fingerprint.

### 2. No Hardcoded Paths or Secrets

❌ **Don't** hardcode file paths:
```python
path = "/Users/ewan/Documents/Stocks/All Options/bot-eod-report-2025-05-07.parquet"
```

✅ **Do** use environment variables or `shared.data_loader`:
```python
from shared.data_loader import load_data, get_latest_file

# Option 1: Use load_data (recommended)
df = load_data("options")

# Option 2: Use environment variable
import os
stocks_dir = Path(os.environ.get("STOCKS_DIR", Path.home() / "Documents" / "Stocks"))
```

❌ **Don't** hardcode API keys:
```python
API_KEY = "sk-proj-abc123def456..."
```

✅ **Do** read from environment:
```python
import os
api_key = os.environ.get("UNUSUAL_WHALES_API_KEY")
if not api_key:
    raise RuntimeError("UNUSUAL_WHALES_API_KEY not set")
```

**Why**: Code gets committed; environment stays local/secure.

### 3. Error Messages Don't Leak Data

❌ **Don't** expose raw data in error messages:
```python
try:
    result = analyze(df)
except Exception as e:
    return text_result(f"Error: {str(e)}\n{df.head(100)}")  # Dumps data!
```

✅ **Do** return generic, safe error messages:
```python
try:
    result = analyze(df)
except FileNotFoundError:
    return text_result({"error": "Data not available for this date"})
except ValueError as e:
    return text_result({"error": f"Invalid parameter: {str(e)}"})
```

**Why**: Stack traces and raw dataframes reveal structure and values.

### 4. No Timestamps in Results

❌ **Don't** include exact trade times:
```python
return df_to_result(df[["ticker", "time", "volume", "premium"]])
# Returns: {"ticker": "AAPL", "time": "2025-05-07T09:30:45.123Z", ...}
```

✅ **Do** use aggregated dates or remove times:
```python
df["date"] = df["timestamp"].dt.date  # Only the date
result = df.groupby(["ticker", "date"]).agg({"volume": "sum"}).reset_index()
return df_to_result(result)
```

**Why**: Exact timestamps identify institutional behavior (e.g., pre-market accumulation).

### 5. Volume Aggregation

When possible, aggregate small sample sizes to avoid fingerprinting:

❌ **Bad** — reveals exact flow:
```python
return text_result({
    "AAPL": [
        {"premium": 15000, "volume": 50},
        {"premium": 15100, "volume": 25},
        {"premium": 15200, "volume": 75},
    ]
})  # 150 contracts total; size breakdown identifies trader
```

✅ **Good** — aggregate:
```python
return text_result({
    "AAPL": {
        "total_premium": 45300,
        "total_volume": 150,
        "avg_premium": 15100,
    }
})  # Can't reconstruct individual orders
```

**Why**: Protects institutional clients; prevents reverse-engineering of positions.

### 6. Logging: Don't Log Raw Data

❌ **Don't** debug-log trade data:
```python
import logging
logger = logging.getLogger(__name__)

logger.debug(f"Loaded dataframe: {df}")  # Logs all rows!
```

✅ **Do** log metadata, not data:
```python
logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
logger.debug(f"Columns: {df.columns.tolist()}")
```

**Why**: Logs can be accidentally captured, sent to centralized servers, or exposed in crash reports.

### 7. Data Retention

Don't keep raw trade data in memory longer than necessary:

```python
def analyze_trades(date: str) -> dict:
    # Load
    df = load_data("options", date=date, usecols=["ticker", "volume"])
    
    # Analyze (fast)
    result = df.groupby("ticker")["volume"].sum()
    
    # Delete
    del df
    
    # Return (only aggregated result)
    return result.to_dict()
```

**Why**: Reduces exposure window; minimizes memory footprint of sensitive data.

### 8. Input Validation

Always validate user inputs before querying data:

```python
def validate_date(date_str: str) -> bool:
    """Check date format."""
    import re
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))

@server.call_tool(name="my_tool")
async def call_my_tool(date: str | None = None) -> list:
    if date and not validate_date(date):
        raise ValueError(f"Invalid date format: {date}")
    
    # Safe to use date
    df = load_data("options", date=date)
    ...
```

**Why**: Prevents injection attacks; ensures predictable data access patterns.

## Sensitive Data Checklist

Before committing or shipping:

- [ ] No raw trade data in results (aggregated only)
- [ ] No exact timestamps (date only, or time buckets)
- [ ] No hardcoded file paths (use `load_data()` or env vars)
- [ ] No API keys/secrets in code (use `os.environ`)
- [ ] Error messages are generic (no raw data or stack traces)
- [ ] Debug logs don't include data (metadata only)
- [ ] No PII in tool descriptions or parameter names
- [ ] Input validation on all user parameters
- [ ] Data deleted from memory after use

## Example: Secure Tool

```python
import logging
from shared.data_loader import load_data
from shared.server_utils import text_result, df_to_result

logger = logging.getLogger(__name__)

@server.call_tool(name="volume_analysis")
async def call_volume_analysis(date: str | None = None, min_volume: int = 1000) -> list:
    """Analyze aggregate volume by ticker.
    
    Returns aggregated (non-identifiable) volume stats.
    """
    # Validate
    if min_volume < 0 or min_volume > 1000000:
        raise ValueError(f"min_volume out of range: {min_volume}")
    
    try:
        # Load with column pushdown
        logger.info(f"Loading data for date={date}")
        df = load_data(
            "options",
            date=date,
            usecols=["ticker", "volume"]
        )
        
        # Aggregate (removes individual trade fingerprints)
        logger.info(f"Aggregating {len(df)} rows")
        result = df[df["volume"] >= min_volume].groupby("ticker").agg({
            "volume": ["sum", "mean", "count"]
        }).reset_index()
        
        # Clean column names
        result.columns = ["ticker", "total_volume", "avg_volume", "trade_count"]
        
        # Delete raw data
        del df
        
        # Return aggregated result
        return df_to_result(result.head(50))
    
    except FileNotFoundError:
        # Generic error (don't expose path)
        logger.warning(f"No data available for date={date}")
        return text_result({"error": "No data available for this date"})
    except Exception as e:
        # Log the real error internally, but don't expose it
        logger.error(f"Unexpected error in volume_analysis: {e}", exc_info=True)
        return text_result({"error": "An error occurred"})
```

## References

- [common/security.md](../../../.claude/rules/common/security.md) — General security practices
- [data-pipeline.md](data-pipeline.md) — Data access conventions (use load_data(), not raw paths)

---

**Questions?** If you're unsure whether a tool result exposes too much data, ask: "Could someone reverse-engineer institutional positions from this?" If yes, aggregate or remove it.
