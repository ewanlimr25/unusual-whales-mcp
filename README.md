# Unusual Whales MCP Server

An MCP (Model Context Protocol) server for analyzing data exported from [Unusual Whales](https://unusualwhales.com). Drop your CSV/JSON data exports into the `data/` directory and query them through Claude Code.

## Setup

```bash
pip install -e .
```

## Usage

### Add to Claude Code

```bash
claude mcp add --transport stdio unusual-whales -- python -m src.server
```

Or add to `.mcp.json`:

```json
{
  "mcpServers": {
    "unusual-whales": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/unusual-whales-mcp"
    }
  }
}
```

### Adding Data

Place your Unusual Whales CSV or JSON exports into the `data/` directory:

```
data/
  options_flow.csv
  dark_pool.csv
  market_data.csv
```

### Available Tools

- **list_data_files** — List all available data files
- **query_data** — Load, filter, and sort data files
- **describe_data** — Get column info and summary statistics
- **analyze_options_flow** — Analyze unusual options activity
- **analyze_dark_pool** — Analyze dark pool / block trades
- **analyze_market_overview** — Sector and market breadth analysis
