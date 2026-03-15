# Unusual Whales MCP Servers

A suite of MCP (Model Context Protocol) servers for analyzing data exported from [Unusual Whales](https://unusualwhales.com). Each server focuses on a specific data type for lean, efficient operation.

## Setup

```bash
cd unusual-whales-mcp
pip install -e .
```

Data is read from `~/Documents/Stocks/` with this expected structure:

```
~/Documents/Stocks/
├── All Options/          # bot-eod-report-YYYY-MM-DD.csv
├── Dark pool/            # dp-eod-report-YYYY-MM-DD.csv
├── Hot Option Chains/    # hot-chains-YYYY-MM-DD.csv
├── Stock Screener/       # stock-screener-YYYY-MM-DD.csv
└── OI changes/           # chain-oi-changes-YYYY-MM-DD.csv
```

## MCP Servers

### uw-screener — Stock Screener Analysis
```bash
claude mcp add --transport stdio uw-screener -- python -m servers.screener
```
| Tool | Description |
|------|-------------|
| `bullish_bearish_screener` | Rank tickers by net bullish vs bearish premium |
| `iv_rank_screener` | Find tickers with extreme IV rank |
| `put_call_ratio_extremes` | Unusual put/call ratio detection |
| `earnings_catalyst_scanner` | Upcoming earnings + elevated IV |
| `volume_vs_average` | Volume spikes vs 30-day average |

### uw-options — All Options Flow Analysis
```bash
claude mcp add --transport stdio uw-options -- python -m servers.options_flow
```
| Tool | Description |
|------|-------------|
| `top_premium_trades` | Largest single-trade bets by premium |
| `unusual_volume_scanner` | Volume >> open interest detection |
| `sweep_detector` | Aggressive ask/bid side sweeps |
| `iv_outliers` | Unusually high implied volatility |
| `sector_flow_summary` | Sector-level premium flow |
| `expiry_heatmap` | Premium concentration by expiry |
| `greek_screener` | Filter by delta/gamma/vega |

### uw-darkpool — Dark Pool Analysis
```bash
claude mcp add --transport stdio uw-darkpool -- python -m servers.dark_pool
```
| Tool | Description |
|------|-------------|
| `largest_dark_pool_trades` | Top block trades with NBBO context |
| `dark_pool_ticker_summary` | Aggregate dark pool per ticker |
| `dark_pool_price_levels` | Institutional support/resistance zones |
| `extended_hours_filter` | Pre/post-market block trades |

### uw-hotchains — Hot Option Chains Analysis
```bash
claude mcp add --transport stdio uw-hotchains -- python -m servers.hot_chains
```
| Tool | Description |
|------|-------------|
| `most_active_contracts` | Hottest contracts by volume/premium |
| `sweep_ratio_scanner` | High sweep-to-volume ratio contracts |
| `smart_money_flow` | Ask vs bid side volume analysis |
| `multileg_activity` | Complex strategy detection |

### uw-oi — Open Interest Changes Analysis
```bash
claude mcp add --transport stdio uw-oi -- python -m servers.oi_changes
```
| Tool | Description |
|------|-------------|
| `biggest_oi_increases` | Largest new position openings |
| `oi_decrease_with_volume` | Position closings detection |
| `smart_positioning` | Bullish/bearish inference from OI + side |

### uw-insights — Cross-Dataset + Yahoo Finance
```bash
claude mcp add --transport stdio uw-insights -- python -m servers.insights
```
| Tool | Description |
|------|-------------|
| `stock_deep_dive` | Full ticker analysis (Yahoo + all UW data) |
| `earnings_play_analyzer` | Pre-earnings setups with OI positioning |
| `price_vs_flow_divergence` | Price action vs options flow disagreement |
| `institutional_accumulation_detector` | Dark pool buy/sell imbalance detection |
| `analyst_vs_flow` | Wall Street analysts vs options traders |
