# MCP Server Patterns

> Patterns specific to writing MCP servers for unusual-whales-mcp. Extends [../../../.claude/rules/common/patterns.md](../../../.claude/rules/common/patterns.md).

## The Standard MCP Server Structure

Every server in unusual-whales-mcp follows this exact pattern. Use this as a checklist when writing or reviewing a new server.

```python
"""MCP Server: <Description>."""

import asyncio
import sys
from pathlib import Path

# Allow running as module from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool
from mcp.server.models import InitializationOptions

from shared.data_loader import load_data, list_available_dates
from shared.server_utils import text_result, df_to_result

# Create server instance with short, lowercase name
server = Server("uw-screener")


# ============================================================================
# STEP 1: Declare all tools available in this server
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all tools provided by this server.
    
    Each Tool has:
    - name: Unique identifier (snake_case)
    - description: 1-2 sentence explanation of what it does and when to use it
    - inputSchema: JSON schema of input parameters (always "object" type)
    """
    return [
        Tool(
            name="bullish_bearish_screener",
            description=(
                "Rank tickers by net bullish vs bearish premium flow. "
                "Use this to find which stocks have the most aggressive directional bets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results", "default": 20},
                    "direction": {
                        "type": "string",
                        "enum": ["bullish", "bearish"],
                        "description": "Filter for most bullish or bearish",
                    },
                },
                "required": [],
            },
        ),
        # ... more tools ...
    ]


# ============================================================================
# STEP 2: Implement each tool as a separate function
# ============================================================================

@server.call_tool(name="bullish_bearish_screener")
async def call_bullish_bearish_screener(
    date: str | None = None,
    top_n: int = 20,
    direction: str = "bullish",
) -> list:
    """Analyze bullish vs bearish flow.
    
    Args:
        date: YYYY-MM-DD or None for latest
        top_n: Number of tickers to return
        direction: "bullish" or "bearish"
    
    Returns:
        list[TextContent] with JSON-formatted results
    
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If no data available for date
    """
    # Validate inputs
    if direction not in ["bullish", "bearish"]:
        raise ValueError(f"direction must be 'bullish' or 'bearish', got '{direction}'")
    
    if top_n < 1 or top_n > 500:
        raise ValueError(f"top_n must be 1-500, got {top_n}")
    
    try:
        # Load data with column pushdown
        df = load_data("options", date=date, usecols=["ticker", "call_volume", "put_volume"])
        
        # Analysis logic
        df["net_bullish"] = df["call_volume"] - df["put_volume"]
        
        if direction == "bullish":
            result = df.nlargest(top_n, "net_bullish")
        else:
            result = df.nsmallest(top_n, "net_bullish")
        
        return df_to_result(result, max_rows=top_n)
    
    except FileNotFoundError as e:
        # Return user-friendly error (don't expose raw paths)
        return text_result({"error": "No data available for this date"})


# ============================================================================
# STEP 3: Server entry point
# ============================================================================

async def main():
    """Start the server via stdio transport."""
    async with stdio_server(server) as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(server_name="uw-screener", server_version="0.3.0"),
        )


if __name__ == "__main__":
    asyncio.run(main())
```

## Key Rules

### 1. Server Name Convention

- Use lowercase, hyphenated names: `uw-screener`, `uw-darkpool`, `uw-options`
- In code: `server = Server("uw-screener")`
- In `pyproject.toml`: `uw-screener = "servers.screener:main"`
- Pattern: `uw-<focus-area>`

### 2. Tool Names

- Use `snake_case`: `bullish_bearish_screener`, `institutional_accumulation_detector`
- Descriptive: "screener" vs "analyze"; "detector" vs "find"
- **No abbreviations** that would confuse the user
- Group related tools in the same server (don't scatter across 10 servers)

### 3. Tool Descriptions

Write 1-2 sentence descriptions that explain:
- **What it does** (noun phrase)
- **When to use it** (trading context or analysis goal)
- **Key output** (optional, if non-obvious)

**Good**:
```
"Rank tickers by net bullish vs bearish premium flow. "
"Use this to find which stocks have the most aggressive directional bets."
```

**Bad**:
```
"Analyzes premium flow."  # Vague, no context
"Bullish bearish screener"  # Not a sentence
```

### 4. Input Schema

- **Always** use `"type": "object"`
- Define every parameter in `properties`
- Include `description` for each parameter with units/format
- Mark required params in `required` (usually empty for optional params)
- Use `default` for optional params with sensible defaults

**Pattern**:
```python
inputSchema={
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": "Date in YYYY-MM-DD format. Defaults to latest."
        },
        "top_n": {
            "type": "integer",
            "description": "Number of results (1-500)",
            "default": 20
        },
        "min_premium": {
            "type": "number",
            "description": "Minimum total premium in dollars (filters noise)",
        },
    },
    "required": [],  # All optional
}
```

### 5. Result Formatting

**Always** use `text_result()` or `df_to_result()` from `shared.server_utils`:

```python
# For dictionaries/structured data
return text_result({"status": "ok", "count": 5, "tickers": ["AAPL", "TSLA"]})

# For DataFrames (auto-caps at 50 rows)
return df_to_result(df, max_rows=20)

# For strings
return text_result("No data available")

# For errors (return as text, not raising)
return text_result({"error": "Invalid date format"})
```

**Why**: Consistent formatting across all servers; auto-pagination for large results.

### 6. Error Handling

**Raise** for validation errors:
```python
if not isinstance(top_n, int) or top_n < 1:
    raise ValueError(f"top_n must be a positive integer, got {top_n}")
```

**Catch and return** for data unavailability:
```python
try:
    df = load_data("options", date=date)
except FileNotFoundError:
    return text_result({"error": "No data available for this date"})
```

**Never** expose internal paths, raw exception messages, or stack traces in results.

### 7. Type Hints

Use Python type hints on **all** function signatures:

```python
async def call_bullish_bearish_screener(
    date: str | None = None,
    top_n: int = 20,
    direction: str = "bullish",
) -> list:
    ...
```

This enables IDE autocomplete and helps catch bugs early.

### 8. Function Size

Keep tool implementation functions **under 100 lines**. If longer, extract analysis logic into `shared/`:

```
shared/
├── analysis_darkpool.py     # Dark pool analysis functions
├── analysis_screener.py     # Stock screener functions
└── ...
```

**Pattern**:
```python
# servers/screener.py
from shared.analysis_screener import rank_bullish_bearish

@server.call_tool(name="bullish_bearish_screener")
async def call_bullish_bearish_screener(...) -> list:
    df = load_data("options", date=date)
    result = rank_bullish_bearish(df, direction, top_n)
    return df_to_result(result)
```

### 9. Date Handling

Always support an optional `date` parameter (YYYY-MM-DD) with sensible default:

```python
"date": {
    "type": "string",
    "description": "Date (YYYY-MM-DD). Defaults to latest.",
}
```

Implementation:
```python
df = load_data("options", date=date)  # date=None → uses latest
```

No custom date parsing; `load_data()` handles it.

### 10. Data Access Pattern

Use `load_data()` with column pushdown for performance:

```python
from shared.data_loader import load_data, get_latest_file
import duckdb

# Simple case: load specific columns
df = load_data("options", date=date, usecols=["ticker", "volume", "premium"])

# Complex case: use DuckDB directly
path = get_latest_file("darkpool")
df = duckdb.query(f"""
    SELECT ticker, SUM(volume) as total_volume
    FROM read_parquet('{path}')
    WHERE premium > 1000
    GROUP BY ticker
    ORDER BY total_volume DESC
    LIMIT {top_n}
""").df()
```

**Why**: Parquet + DuckDB = efficient, only loads needed columns.

## Testing MCP Servers

Use pytest with markers:

```python
# tests/test_screener.py
import pytest
from servers.screener import call_bullish_bearish_screener

@pytest.mark.asyncio
@pytest.mark.unit
async def test_bullish_bearish_screener_valid():
    result = await call_bullish_bearish_screener(top_n=5, direction="bullish")
    assert len(result) == 1
    assert "records" in result[0].text

@pytest.mark.asyncio
@pytest.mark.unit
async def test_bullish_bearish_screener_invalid_direction():
    with pytest.raises(ValueError, match="direction must be"):
        await call_bullish_bearish_screener(direction="sideways")

@pytest.mark.asyncio
@pytest.mark.integration
async def test_bullish_bearish_screener_with_data():
    # Requires actual Parquet files in ~/Documents/Stocks/
    result = await call_bullish_bearish_screener(top_n=10)
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert len(data) <= 10
```

## Checklist for New Servers

- [ ] Server name follows `uw-<focus>` convention
- [ ] All tools documented with descriptions
- [ ] All inputs validated with clear error messages
- [ ] Results use `text_result()` or `df_to_result()`
- [ ] Type hints on all functions
- [ ] Analysis logic extracted to `shared/` if >50 lines per tool
- [ ] Entry point: `async def main()` + `if __name__ == "__main__"`
- [ ] Tests: unit tests + integration tests if data-dependent
- [ ] No hardcoded paths; use `load_data()` or environment variables
- [ ] No prints; use logging if needed (Python logging module)

---

**Related**: See [data-pipeline.md](data-pipeline.md) for DuckDB/Parquet conventions and [financial-security.md](financial-security.md) for sensitive data handling.
