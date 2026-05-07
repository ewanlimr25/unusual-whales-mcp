# CLAUDE.md — Unusual Whales MCP

## Overview

**unusual-whales-mcp** is a suite of 11 MCP (Model Context Protocol) servers that analyze financial data exported from [Unusual Whales](https://unusualwhales.com). Each server is a lean, independent tool that reads Parquet files via DuckDB and exposes specialized analysis functions.

- **Language**: Python 3.12+
- **Package Manager**: uv, hatchling
- **Data Format**: Parquet (converted from CSV via `convert.py`)
- **Data Access**: DuckDB for efficient querying
- **Testing**: pytest (80%+ coverage required)

## Architecture

### Core Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `servers/` | 10 independent MCP servers, each focused on a data type or analysis | Production |
| `shared/` | Data loader, server utilities, common functions | Stable |
| `convert.py` | CSV → Parquet conversion tool | Stable |
| `data/` | Sample/test data | Development |

### The 11 Servers

```
uw-screener           → Stock Screener analysis (bullish/bearish flows, IV rank)
uw-options-flow       → Trade-level options flow (sweeps, premium, vol/OI)
uw-options-structure  → Per-ticker structural (IV term, skew, GEX, DEX)
uw-darkpool           → Dark Pool analysis (accumulation, conviction)
uw-hotchains          → Hot Option Chains (unusual chains)
uw-oi                 → Open Interest changes (OI buildup, OPEX concentration)
uw-insights           → Cross-data synthesis (deep-dive, conviction, signal_confluence)
uw-historical         → Historical data lookups (price, volatility)
uw-watchlist          → Watchlist management (save, filter, track)
uw-playbook           → Strategy suggestions + daily_synthesis morning briefing
uw-risk               → Portfolio correlation + market regime
```

### Data Storage

```
~/Documents/Stocks/
├── All Options/          → bot-eod-report-*.parquet
├── Dark pool/            → dp-eod-report-*.parquet
├── Hot Option Chains/    → hot-chains-*.parquet
├── Stock Screener/       → stock-screener-*.parquet
└── OI changes/           → chain-oi-changes-*.parquet
```

The data loader automatically discovers the latest file or accepts a date parameter.

## Daily Workflow

### 1. Setup (one-time)

```bash
pip install -e .
python convert.py  # Convert existing CSVs to Parquet
```

### 2. Register Servers (once, global)

```bash
PROJECT_DIR="$HOME/Development/unusual-whales-mcp"
declare -A UW_SERVERS=(
  [uw-screener]=screener
  [uw-options-flow]=options_flow
  [uw-options-structure]=options_structure
  [uw-darkpool]=dark_pool
  [uw-hotchains]=hot_chains
  [uw-oi]=oi_changes
  [uw-insights]=insights
  [uw-historical]=historical
  [uw-watchlist]=watchlist
  [uw-playbook]=playbook
  [uw-risk]=risk
)
for name in "${!UW_SERVERS[@]}"; do
  module="${UW_SERVERS[$name]}"
  claude mcp add --transport stdio --scope user "$name" -- \
    bash -c "cd $PROJECT_DIR && python -m servers.$module"
done
```

### 3. Daily Data Refresh

```bash
# Download CSVs from Unusual Whales → ~/Documents/Stocks/<subfolder>
python convert.py  # Safe to run daily; skips already-converted files
```

### 4. Query Workflow

```
Market Pulse (2 min):
  → "Give me my morning briefing"          (uw-insights: daily_synthesis)
  → "What's the market regime?"            (uw-insights: market_regime)
  → "Which tickers are most bullish?"      (uw-screener: bullish_bearish_screener)

Unusual Activity Hunt (5 min):
  → "Show me the biggest sweeps"           (uw-options: sweep_ratio_scanner)
  → "Where is OI building the most?"       (uw-oi: biggest_oi_increases)

Deep Dives (5-10 min):
  → "Do a deep dive on AAPL"               (uw-strategy: stock_deep_dive)
  → "Is there dark pool accumulation?"     (uw-darkpool: institutional_accumulation_detector)
```

## Development Standards

### Code Organization

- **Files**: Each server is a single module under `servers/`. Keep to **<500 lines**.
- **Shared utilities**: Place reusable logic in `shared/`.
- **Data functions**: Put analysis logic in a separate file (e.g., `shared/analysis_darkpool.py`) if it grows beyond 100 lines.

### MCP Server Pattern

Every server:

1. Imports and initializes a `Server` instance (e.g., `server = Server("uw-screener")`)
2. Implements `@server.list_tools()` returning a list of `Tool` objects with descriptions
3. Implements `@server.call_tool(name, arguments)` dispatching to tool functions
4. Defines tool functions that return data via `text_result()` or `df_to_result()`
5. Ends with `stdio_server()` to start the server

**Example structure**:
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool
from shared.server_utils import text_result, df_to_result

server = Server("uw-example")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="my_tool", description="...", inputSchema={...}),
    ]

@server.call_tool(name="my_tool")
async def call_my_tool(param1: str, param2: int = 10) -> list:
    # Tool implementation
    return text_result(result)

async def main():
    async with stdio_server(server) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, InitializationOptions(...))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Data Access Pattern

Use `shared.data_loader`:

```python
from shared.data_loader import load_data, list_available_dates, get_latest_file

# Load with column pushdown + row limit
df = load_data("options", date="2025-05-07", usecols=["ticker", "volume"], nrows=100)

# Or use DuckDB directly for complex queries:
import duckdb
path = get_latest_file("darkpool")
result = duckdb.query(f"""
    SELECT ticker, SUM(volume) as total_volume
    FROM read_parquet('{path}')
    WHERE premium > 1000
    GROUP BY ticker
    ORDER BY total_volume DESC
    LIMIT 20
""").df()
```

### Result Formatting

All tool results go through `text_result()` or `df_to_result()`:

```python
# For structured data
return text_result({"status": "ok", "tickers": ["AAPL", "TSLA"]})

# For DataFrames (auto-caps at 50 rows)
return df_to_result(df, max_rows=20)

# For custom JSON
return text_result(df.to_json(orient="records", indent=2, default_handler=str))
```

### Error Handling

Raise `ValueError` or `FileNotFoundError` with clear messages:

```python
if date and not is_valid_date(date):
    raise ValueError(f"Invalid date format: {date}. Use YYYY-MM-DD.")

try:
    df = load_data("options", date=date, usecols=cols)
except FileNotFoundError as e:
    return text_result({"error": str(e)})
```

## Testing

- **Framework**: pytest
- **Coverage**: 80%+ required
- **Types**: Run `mypy --strict` on edited files

```bash
pytest --cov=servers --cov=shared --cov-report=term-missing

# Type checking
mypy --strict servers/ shared/
```

## Security & Data Sensitivity

**Critical**: This project handles **trading data** (options flows, dark pool activity, institutional positions). Treat with the same care as user PII.

- ✅ **Do**: Validate all inputs; sanitize output if exposing to external APIs
- ✅ **Do**: Log at INFO level (not DEBUG) to avoid dumping raw trades
- ❌ **Don't**: Hardcode API keys, tokens, or file paths
- ❌ **Don't**: Expose raw trade details in error messages
- ❌ **Don't**: Cache sensitive data in memory longer than necessary

Use environment variables for paths:
```python
import os
STOCKS_DIR = Path(os.environ.get("STOCKS_DIR", Path.home() / "Documents" / "Stocks"))
```

## Dependencies

| Package | Role | Version |
|---------|------|---------|
| mcp | Model Context Protocol SDK | >=1.0.0 |
| pandas | Data manipulation | >=2.0.0 |
| duckdb | Columnar analytics + Parquet | >=1.0.0 |
| yfinance | Free baseline stock data | >=0.2.0 |

All in `pyproject.toml` under `[project.dependencies]`.

## Common Tasks

### Add a New Server

1. Create `servers/my_analyzer.py` following the MCP server pattern (see above)
2. Add entry point in `pyproject.toml`: `uw-my-analyzer = "servers.my_analyzer:main"`
3. Register with Claude Code:
   ```bash
   claude mcp add --transport stdio --scope user uw-my-analyzer -- \
     bash -c "cd ~/Development/unusual-whales-mcp && python -m servers.my_analyzer"
   ```
4. Add tests in `tests/test_my_analyzer.py`

### Add a New Tool to an Existing Server

1. Define a `Tool` object in `@server.list_tools()`
2. Implement the tool function in `@server.call_tool()`
3. Test locally: `python -m servers.screener` (won't start; just checks imports)
4. Add a unit test for the tool logic

### Debug a Server

```bash
# Run the server and see what it exports
python -m servers.screener

# Run with debug logging
DEBUG=1 python -m servers.screener
```

### Convert New Data

```bash
# After downloading CSVs from Unusual Whales
python convert.py

# Verify conversion
ls -lh ~/Documents/Stocks/*/*.parquet
```

## Commits & PRs

Follow the common git workflow (`~/.claude/rules/common/git-workflow.md`):

- **Format**: `<type>: <description>` (feat, fix, refactor, docs, test, chore)
- **Scope**: `feat(screener): add new iv_rank_screener tool`
- **Body**: Explain **why**, not what

Example:
```
feat(darkpool): add institutional_accumulation_detector tool

Detect quiet accumulation by analyzing dark pool volume across multiple
dates. Helps identify institutional accumulation before price moves.
Addresses feedback from trader analysis workflow.
```

## Helpful Resources

- **MCP Protocol**: https://modelcontextprotocol.io
- **DuckDB Docs**: https://duckdb.org/docs/
- **Pandas**: https://pandas.pydata.org/docs/
- **Unusual Whales**: https://unusualwhales.com/

## Project Status

- **Version**: 0.4.0
- **Stability**: 11 servers, 61 tools in active use
- **Next**: v0.5.0 — remove `put_call_ratio_extremes`, run naming-cleanup pass (audit §4.3), evaluate honourable-mention tools (`multi_day_smart_money_flow`, `0dte_call_put_imbalance`, `realized_vs_implied_move`)

---

**Questions?** Check `.claude/rules/` for extended coding standards, data pipeline conventions, and financial data security guidelines.
