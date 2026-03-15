# Unusual Whales MCP Servers

A suite of 10 MCP (Model Context Protocol) servers for analyzing data exported from [Unusual Whales](https://unusualwhales.com). Each server focuses on a specific data type or analysis function for lean, efficient operation.

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

Download your daily data from Unusual Whales and drop the CSVs into the corresponding folders. The tools automatically use the most recent file, or you can specify a date.

## Adding to Claude Code

Register all servers globally (available from any directory):

```bash
PROJECT_DIR="$HOME/Development/unusual-whales-mcp"
claude mcp add --transport stdio --scope user uw-screener -- bash -c "cd $PROJECT_DIR && python -m servers.screener"
claude mcp add --transport stdio --scope user uw-options -- bash -c "cd $PROJECT_DIR && python -m servers.options_flow"
claude mcp add --transport stdio --scope user uw-darkpool -- bash -c "cd $PROJECT_DIR && python -m servers.dark_pool"
claude mcp add --transport stdio --scope user uw-hotchains -- bash -c "cd $PROJECT_DIR && python -m servers.hot_chains"
claude mcp add --transport stdio --scope user uw-oi -- bash -c "cd $PROJECT_DIR && python -m servers.oi_changes"
claude mcp add --transport stdio --scope user uw-insights -- bash -c "cd $PROJECT_DIR && python -m servers.insights"
claude mcp add --transport stdio --scope user uw-historical -- bash -c "cd $PROJECT_DIR && python -m servers.historical"
claude mcp add --transport stdio --scope user uw-watchlist -- bash -c "cd $PROJECT_DIR && python -m servers.watchlist"
claude mcp add --transport stdio --scope user uw-strategy -- bash -c "cd $PROJECT_DIR && python -m servers.strategy"
claude mcp add --transport stdio --scope user uw-risk -- bash -c "cd $PROJECT_DIR && python -m servers.risk"
```

Also recommended — add Yahoo Finance for free baseline stock data:
```bash
claude mcp add --transport stdio --scope user yahoo-finance -- uvx mcp-yahoo-finance
```

---

## Daily Workflow

### Step 1: Download Data
At end of day, download your CSVs from Unusual Whales and drop them into `~/Documents/Stocks/` subfolders.

### Step 2: Market Pulse (2 min)
Get the big picture before drilling in.

```
"What's the market regime right now?"              → market_regime
"Show me the sector flow summary"                  → sector_flow_summary
"Which tickers have the most bullish flow today?"   → bullish_bearish_screener
"Top dark pool tickers by premium"                  → dark_pool_ticker_summary
```

### Step 3: Unusual Activity Hunt (5 min)
Find what's abnormal.

```
"Which tickers spiked in volume vs their average?"  → volume_vs_average
"Show me the biggest sweeps today"                  → sweep_ratio_scanner
"Where is OI building the most?"                    → biggest_oi_increases
"Find tickers with the highest confluence score"    → signal_confluence
```

### Step 4: Deep Dives (5-10 min)
Investigate flagged tickers.

```
"Do a deep dive on AAPL"                           → stock_deep_dive
"Is there dark pool accumulation in TSLA?"          → institutional_accumulation_detector
"Do analysts agree with the options flow on NVDA?"  → analyst_vs_flow
"Is the flow diverging from price on META?"         → price_vs_flow_divergence
```

### Step 5: Trade Ideas (5 min)
Turn signals into strategies.

```
"What strategy should I use on AAPL?"               → suggest_strategy
"Scan my watchlist for strategies"                   → batch_strategy_scan
"Find earnings plays in the next 2 weeks"            → earnings_play_analyzer
"What are the best bullish setups with 3+ signals?"  → signal_confluence
```

### Step 6: Risk Check (2 min)
Before committing capital.

```
"Check correlation of AAPL, MSFT, GOOGL, AMZN"     → portfolio_correlation
"Any alerts on my watchlist?"                        → watchlist_alerts
```

---

## MCP Servers Reference

### uw-screener — Stock Screener Analysis
**When to use:** Start of your daily scan. Broad market overview of 6K+ tickers.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `bullish_bearish_screener` | Rank tickers by net bullish vs bearish premium | Start here — see where money is flowing |
| `iv_rank_screener` | Find tickers with extreme IV rank | Finding premium selling (high IV) or buying (low IV) candidates |
| `put_call_ratio_extremes` | Unusual put/call ratio detection | Spotting hedging or aggressive directional positioning |
| `earnings_catalyst_scanner` | Upcoming earnings + elevated IV | Finding pre-earnings setups |
| `volume_vs_average` | Volume spikes vs 30-day average | Detecting unusual activity that breaks the pattern |

### uw-options — All Options Flow Analysis
**When to use:** Deep-diving into the raw options tape (9.8M trades). Slow on first load but very detailed.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `top_premium_trades` | Largest single-trade bets by premium | Finding the whale bets |
| `unusual_volume_scanner` | Volume >> open interest detection | Spotting brand new positions being opened |
| `sweep_detector` | Aggressive ask/bid side sweeps | Finding urgency — someone sweeping means they need in NOW |
| `iv_outliers` | Unusually high implied volatility | Spotting expected big moves or mispricing |
| `sector_flow_summary` | Sector-level premium flow | Understanding macro rotation (risk-on vs defensive) |
| `expiry_heatmap` | Premium concentration by expiry | Seeing if bets are short-term (weeklies) or longer-term |
| `greek_screener` | Filter by delta/gamma/vega | Finding specific trade profiles (directional, vol bets, etc.) |

### uw-darkpool — Dark Pool Analysis
**When to use:** Understanding institutional activity. Dark pool = where the big money trades quietly.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `largest_dark_pool_trades` | Top block trades with NBBO context | Finding the biggest institutional moves |
| `dark_pool_ticker_summary` | Aggregate dark pool per ticker | Ranking tickers by institutional interest |
| `dark_pool_price_levels` | Institutional support/resistance zones | Finding where institutions are building positions |
| `extended_hours_filter` | Pre/post-market block trades | Catching early signals before the next session |

### uw-hotchains — Hot Option Chains Analysis
**When to use:** Per-contract analysis. More aggregated than raw options flow, faster to scan.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `most_active_contracts` | Hottest contracts by volume/premium | Finding the day's most-traded contracts |
| `sweep_ratio_scanner` | High sweep-to-volume ratio contracts | Aggressive directional bets |
| `smart_money_flow` | Ask vs bid side volume analysis | Gauging if buyers or sellers are more aggressive |
| `multileg_activity` | Complex strategy detection | Spotting institutional spreads and combos |

### uw-oi — Open Interest Changes Analysis
**When to use:** Tracking position building over time. OI changes tell you about commitment, not just noise.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `biggest_oi_increases` | Largest new position openings | Finding fresh conviction (new money, not day-trading) |
| `oi_decrease_with_volume` | Position closings detection | Spotting profit-taking or capitulation |
| `smart_positioning` | Bullish/bearish inference from OI + side | Determining if new positions are bullish or bearish |

### uw-insights — Cross-Dataset + Yahoo Finance
**When to use:** Combining Unusual Whales signals with Yahoo Finance fundamentals for a complete picture.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `stock_deep_dive` | Full ticker analysis (Yahoo + all UW data) | Complete 360° view of a single ticker |
| `earnings_play_analyzer` | Pre-earnings setups with OI positioning | Finding earnings trades with unusual pre-positioning |
| `price_vs_flow_divergence` | Price action vs options flow disagreement | When smart money disagrees with price — reversal signal |
| `institutional_accumulation_detector` | Dark pool buy/sell imbalance detection | Catching stealth accumulation before price moves |
| `analyst_vs_flow` | Wall Street analysts vs options traders | When traders bet against Wall Street — who's right? |

### uw-historical — Historical Trends & Backtesting
**When to use:** Comparing across multiple days. Needs accumulated data (download daily to build history).

| Tool | Description | When to Use |
|------|-------------|-------------|
| `available_dates` | List all data dates you have | Check what history is available |
| `trend_analyzer` | Multi-day trend for a ticker's metrics | "Is AAPL's bullish flow a one-day blip or a multi-day trend?" |
| `oi_trend` | OI buildup/decline over multiple days | "Are positions steadily building in this name?" |
| `signal_backtest` | Past signal → price outcome verification | "When this signal fired before, did the stock actually move?" |

### uw-watchlist — Watchlist Management
**When to use:** After you've identified your focus tickers. Cuts through 6K tickers to show only what you care about.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `manage_watchlist` | Add/remove/list tickers and groups | Build your watchlist (e.g., "earnings_plays", "momentum") |
| `watchlist_scan` | Comprehensive scan on watchlist only | Daily check on your tracked tickers |
| `watchlist_alerts` | Flag only unusual activity on watchlist | "Did anything notable happen on my tickers today?" |

### uw-strategy — Strategy Suggestions
**When to use:** You found an interesting ticker and want to know HOW to trade it.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `suggest_strategy` | Multi-factor analysis → strategy suggestion | "What options strategy fits AAPL's current signals?" |
| `batch_strategy_scan` | Analyze multiple tickers for trade ideas | "Scan my watchlist — which tickers have a trade setup?" |

### uw-risk — Risk & Market Regime
**When to use:** Before placing trades. Ensures you're not overconcentrated and trading with the macro trend.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `portfolio_correlation` | Sector concentration and price correlation | "Are my 5 trade ideas actually 1 correlated bet?" |
| `market_regime` | SPY/VIX trend + breadth classification | "Should I be aggressive or defensive right now?" |
| `signal_confluence` | Multi-factor scoring across all tickers | "Show me tickers with 4+ bullish signals aligning" |

---

## Example Prompts

**Morning scan:**
> "Check the market regime. Then show me the top 10 bullish tickers by signal confluence. Do a deep dive on the top 3."

**Earnings week:**
> "Find all earnings plays in the next 7 days with IV rank above 60. Suggest strategies for each."

**Watchlist check:**
> "Any alerts on my watchlist? For any ticker with alerts, suggest a strategy."

**Risk check:**
> "I'm looking at AAPL, MSFT, GOOGL, AMZN, and META. Check the correlation and tell me if I'm too concentrated."

**Historical analysis:**
> "Show me the trend for TSLA over the last 5 days. Is OI building or unwinding?"

**Full workflow:**
> "Run my end-of-day analysis: market regime, top confluence tickers, watchlist alerts, and suggest strategies for anything interesting."
