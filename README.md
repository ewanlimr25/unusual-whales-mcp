# Unusual Whales MCP Servers

A suite of 11 MCP (Model Context Protocol) servers and 61 tools for analyzing data exported from [Unusual Whales](https://unusualwhales.com). Each server focuses on a specific data type or analysis function for lean, efficient operation.

> ## ⚠ v0.4.0 Migration
>
> The server layout changed in v0.4.0 (Phase 2 of the refactor). If you previously
> registered `uw-options` or `uw-strategy`, you must re-register:
>
> ```bash
> claude mcp remove uw-options
> claude mcp remove uw-strategy
>
> PROJECT_DIR="$HOME/Development/unusual-whales-mcp"
> claude mcp add --transport stdio --scope user uw-options-flow      -- bash -c "cd $PROJECT_DIR && python -m servers.options_flow"
> claude mcp add --transport stdio --scope user uw-options-structure -- bash -c "cd $PROJECT_DIR && python -m servers.options_structure"
> claude mcp add --transport stdio --scope user uw-playbook          -- bash -c "cd $PROJECT_DIR && python -m servers.playbook"
> ```
>
> What moved:
> - `uw-options` (14 tools) → split into **`uw-options-flow`** (9 trade-level tools) + **`uw-options-structure`** (7 per-ticker structural tools including new `dealer_delta_exposure` and `vanna_charm_exposure`).
> - `uw-strategy` (2 tools) → renamed to **`uw-playbook`** and absorbs `daily_synthesis` (moved from `uw-insights`).
> - `signal_confluence` moved from `uw-risk` to `uw-insights` (no re-registration needed; just a different server hosts it).
> - `gamma_exposure_profile` now defaults to `dte_max=45` (was: all DTEs) to avoid LEAP bias on the zero-gamma level. Pass `dte_max=365` to restore the old behaviour.
>
> New tools in v0.4.0:
> - `dp_block_size_stratified` (uw-darkpool) — premium tier breakdown
> - `opex_concentration` (uw-oi) — per-ticker OI concentration on a single expiry
> - `dealer_delta_exposure` (uw-options-structure) — DEX
> - `iv_percentile_zscore` (uw-historical) — outlier-robust IV ranking
> - `gex_time_series` (uw-historical) — multi-day ZGL trajectory
> - `volatility_risk_premium` (uw-historical) — IV − realised vol
> - `vanna_charm_exposure` (uw-options-structure) — vanna-squeeze setup detection
> - `pin_risk_screener` (uw-oi) — OPEX-week pinning candidates
>
> Deprecations:
> - `put_call_ratio_extremes` (uw-screener) — use `pc_ratio_zscore` instead. Removed in v0.5.0.
>
> Bug fixes:
> - OPRA option-type parser is now position-aware (was: `str.contains("C")` misclassified MCD/KC/CF puts as calls).
> - `signal_backtest` lookback now uses a generous calendar buffer; non-directional signals (`high_iv_rank`, `volume_spike`) report `vol_realisation_rate` instead of `win_rate`.

## Setup

```bash
cd unusual-whales-mcp
pip install -e .
```

Data is read from `~/Documents/Stocks/`. The MCP tools read **Parquet files** — run `convert.py` once after downloading CSVs to convert them (see [Daily Workflow](#daily-workflow)).

```
~/Documents/Stocks/
├── All Options/          # bot-eod-report-YYYY-MM-DD.parquet
├── Dark pool/            # dp-eod-report-YYYY-MM-DD.parquet
├── Hot Option Chains/    # hot-chains-YYYY-MM-DD.parquet
├── Stock Screener/       # stock-screener-YYYY-MM-DD.parquet
└── OI changes/           # chain-oi-changes-YYYY-MM-DD.parquet
```

The tools automatically use the most recent file, or you can specify a date.

### Converting CSVs to Parquet

```bash
python convert.py           # convert new CSVs → Parquet (deletes CSVs)
python convert.py --revert  # roll back: Parquet → CSV (deletes Parquets)
```

- Already-converted files are skipped, so re-running daily is safe.
- Parquet files are ~5–10x smaller than CSVs and load significantly faster via DuckDB.

## Adding to Claude Code

Register all servers globally (available from any directory):

```bash
PROJECT_DIR="$HOME/Development/unusual-whales-mcp"
claude mcp add --transport stdio --scope user uw-screener          -- bash -c "cd $PROJECT_DIR && python -m servers.screener"
claude mcp add --transport stdio --scope user uw-options-flow      -- bash -c "cd $PROJECT_DIR && python -m servers.options_flow"
claude mcp add --transport stdio --scope user uw-options-structure -- bash -c "cd $PROJECT_DIR && python -m servers.options_structure"
claude mcp add --transport stdio --scope user uw-darkpool          -- bash -c "cd $PROJECT_DIR && python -m servers.dark_pool"
claude mcp add --transport stdio --scope user uw-hotchains         -- bash -c "cd $PROJECT_DIR && python -m servers.hot_chains"
claude mcp add --transport stdio --scope user uw-oi                -- bash -c "cd $PROJECT_DIR && python -m servers.oi_changes"
claude mcp add --transport stdio --scope user uw-insights          -- bash -c "cd $PROJECT_DIR && python -m servers.insights"
claude mcp add --transport stdio --scope user uw-historical        -- bash -c "cd $PROJECT_DIR && python -m servers.historical"
claude mcp add --transport stdio --scope user uw-watchlist         -- bash -c "cd $PROJECT_DIR && python -m servers.watchlist"
claude mcp add --transport stdio --scope user uw-playbook          -- bash -c "cd $PROJECT_DIR && python -m servers.playbook"
claude mcp add --transport stdio --scope user uw-risk              -- bash -c "cd $PROJECT_DIR && python -m servers.risk"
```

Also recommended — add Yahoo Finance for free baseline stock data:
```bash
claude mcp add --transport stdio --scope user yahoo-finance -- uvx mcp-yahoo-finance
```

---

## Daily Workflow

### Step 1: Download & Convert Data
Download your CSVs from Unusual Whales and drop them into `~/Documents/Stocks/` subfolders, then convert:

```bash
python convert.py
```

Already-converted files are skipped — safe to run every day.

### Step 2: Market Pulse (2 min)
Get the big picture before drilling in.

```
"Give me my morning briefing"                       → daily_synthesis  (one call: regime + confluence + watchlist)
"What's the market regime right now?"               → market_regime
"Show me the sector flow summary"                   → sector_flow_summary
"Has sector rotation persisted over the last 5 days?" → sector_flow_persistence
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
"Do a deep dive on AAPL"                                 → stock_deep_dive
"Is there dark pool accumulation in TSLA?"               → institutional_accumulation_detector
"What's the conviction behind TSLA's dark pool flow?"    → conviction_matrix
"Do analysts agree with the options flow on NVDA?"       → analyst_vs_flow
"Is the flow diverging from price on META?"              → price_vs_flow_divergence
"Show me the IV term structure for SPY"                  → iv_term_structure
"What's the zero-gamma level for AAPL today?"            → today_gamma_flip / gamma_exposure_profile
"Where does SPY's dealer delta sit?"                     → dealer_delta_exposure
"Any vanna-squeeze setup on QQQ?"                        → vanna_charm_exposure
"How has SPY's GEX trajectory looked the last 30 days?"  → gex_time_series
"Where is SPY's IV percentile vs the last year?"         → iv_percentile_zscore
"Is SPY's volatility premium attractive for selling?"    → volatility_risk_premium
"Has NVDA's put/call sentiment hit an extreme?"          → pc_ratio_zscore
"Has AAPL been quietly accumulating premium for weeks?"  → cumulative_premium_flow
"Which dark pool prints are mega-block on AAPL?"         → dp_block_size_stratified
"Which OPEX week tickers might pin?"                     → pin_risk_screener
"Is AAPL's OI concentrated on one expiry?"               → opex_concentration
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
| `put_call_ratio_extremes` | **[DEPRECATED v0.4.0]** Use `pc_ratio_zscore` instead — raw P/C extremes for one day | Removed in v0.5.0 |
| `earnings_catalyst_scanner` | Upcoming earnings + elevated IV | Finding pre-earnings setups |
| `volume_vs_average` | Volume spikes vs 30-day average | Detecting unusual activity that breaks the pattern |

### uw-options-flow — Trade-Level Options Flow
**When to use:** Deep-diving into the raw options tape (9.8M trades). Trade-level views — what hit the tape today.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `top_premium_trades` | Largest single-trade bets by premium | Finding the whale bets |
| `unusual_volume_scanner` | Volume >> open interest detection | Spotting brand new positions being opened |
| `sweep_detector` | Aggressive ask/bid side sweeps | Finding urgency — someone sweeping means they need in NOW |
| `iv_outliers` | Unusually high implied volatility | Spotting expected big moves or mispricing |
| `sector_flow_summary` | Sector-level premium flow | Understanding macro rotation (risk-on vs defensive) |
| `expiry_heatmap` | Premium concentration by expiry | Seeing if bets are short-term (weeklies) or longer-term |
| `greek_screener` | Filter by delta/gamma/vega | Finding specific trade profiles (directional, vol bets, etc.) |
| `dte_volume_share` | Volume share by DTE bucket (0DTE / weekly / monthly / LEAP) | Regime hint — high 0DTE = retail session; high monthly+ = institutional |
| `sector_flow_persistence` | Multi-day per-sector net flow + persistence score | Detecting durable sector rotations vs single-day noise |

### uw-options-structure — Per-Ticker Structural Options Tools
**When to use:** Per-symbol structural snapshots — IV term structure, skew, dealer hedging exposure (GEX, DEX).

| Tool | Description | When to Use |
|------|-------------|-------------|
| `iv_term_structure` | IV by expiry — BACKWARDATION / CONTANGO / KINKED shape | Detecting imminent events (backwardation) or binary expiry kinks |
| `term_skew` | Back-month ~25Δ put/call IV skew at a target DTE | Measuring tail-risk demand vs complacency at multi-month horizons |
| `front_end_iv_ratio` | Near-term IV ÷ back-end IV — single tradeable ratio | Quick panic/event detector; ratio > 1.05 = backwardation |
| `today_gamma_flip` | 0DTE zero-gamma level and ATM flip strike for intraday | Precise dealer-hedging map for 0DTE/intraday trading |
| `gamma_exposure_profile` | Net dealer GEX per strike + Zero Gamma Level (default `dte_max=45`) | Finding where dealer hedging amplifies or pins price moves |
| `dealer_delta_exposure` | Net dealer delta hedge requirement (DEX) per ticker | Pre-directional move signal — DEX flips often precede price moves (Karsan / SqueezeMetrics) |
| `vanna_charm_exposure` | Net vanna and charm per ticker — vanna-squeeze setup detection | Put-heavy book + falling VIX = classic vanna-squeeze BUY setup (Karsan/SpotGamma) |

### uw-darkpool — Dark Pool Analysis
**When to use:** Understanding institutional activity. Dark pool = where the big money trades quietly.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `largest_dark_pool_trades` | Top block trades with NBBO context | Finding the biggest institutional moves |
| `dark_pool_ticker_summary` | Aggregate dark pool per ticker | Ranking tickers by institutional interest |
| `dark_pool_price_levels` | Institutional support/resistance zones | Finding where institutions are building positions |
| `extended_hours_filter` | Pre/post-market block trades | Catching early signals before the next session |
| `dp_block_size_stratified` | Premium tier breakdown (mega/block/large/retail) with buy/sell ratio per tier | Filtering retail noise from institutional smart-money signal |

### uw-hotchains — Hot Option Chains Analysis
**When to use:** Per-contract analysis. More aggregated than raw options flow, faster to scan.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `most_active_contracts` | Hottest contracts by volume/premium | Finding the day's most-traded contracts |
| `sweep_ratio_scanner` | High sweep-to-volume ratio contracts | Aggressive directional bets |
| `smart_money_flow` | Ask vs bid side volume analysis | Gauging if buyers or sellers are more aggressive |
| `multi_day_sweep_persistence` | Tickers appearing in top sweeps across multiple sessions | Distinguishing conviction sweep campaigns from single-day news |
| `multileg_activity` | Complex strategy detection | Spotting institutional spreads and combos |

### uw-oi — Open Interest Changes Analysis
**When to use:** Tracking position building over time. OI changes tell you about commitment, not just noise.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `biggest_oi_increases` | Largest new position openings (supports `min_dte`/`max_dte` filters) | Finding fresh conviction — add DTE filters for LEAP-only scans |
| `oi_decrease_with_volume` | Position closings detection | Spotting profit-taking or capitulation |
| `smart_positioning` | Bullish/bearish inference from OI + side | Determining if new positions are bullish or bearish |
| `position_rolling_detector` | Same-day near→far DTE OI roll detection | Catching institutions extending positions rather than closing |
| `opex_concentration` | Per-ticker OI concentration on a single expiry, with pin-distance | OPEX-week pin-risk and dealer-hedging cliffs (Stoll-Whaley / Ni-Pearson-Poteshman) |
| `pin_risk_screener` | Tickers near OPEX ranked by gamma-weighted distance × OI mass | Identifying strongest pinning candidates in OPEX week (Avellaneda-Lipkin 2003) |

### uw-insights — Cross-Dataset + Yahoo Finance
**When to use:** Combining Unusual Whales signals with Yahoo Finance fundamentals for a complete picture.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `stock_deep_dive` | Full ticker analysis (Yahoo + all UW data) | Complete 360° view of a single ticker |
| `earnings_play_analyzer` | Pre-earnings setups with OI positioning | Finding earnings trades with unusual pre-positioning |
| `price_vs_flow_divergence` | Price action vs options flow disagreement | When smart money disagrees with price — reversal signal |
| `institutional_accumulation_detector` | Dark pool buy/sell imbalance detection | Catching stealth accumulation before price moves |
| `analyst_vs_flow` | Wall Street analysts vs options traders | When traders bet against Wall Street — who's right? |
| `conviction_matrix` | Dark pool pressure × options flow → scenario classification | Reveals the "why" — DIRECTIONAL_LONG, HEDGED_LONG, COVERED_CALL, SHORT, or MIXED |
| `signal_confluence` | Multi-factor scoring across all tickers (moved from uw-risk in v0.4.0) | "Show me tickers with 4+ bullish signals aligning" |

### uw-historical — Historical Trends & Backtesting
**When to use:** Comparing across multiple days. Needs accumulated data (download daily to build history).

| Tool | Description | When to Use |
|------|-------------|-------------|
| `available_dates` | List all data dates you have | Check what history is available |
| `trend_analyzer` | Multi-day trend for a ticker's metrics | "Is AAPL's bullish flow a one-day blip or a multi-day trend?" |
| `oi_trend` | OI buildup/decline over multiple days | "Are positions steadily building in this name?" |
| `cumulative_premium_flow` | Sum net directional premium across N sessions (default 90) | LEAP-grade signature — slow multi-week accretion reveals stealth accumulation |
| `pc_ratio_zscore` | Z-score of put/call ratio vs trailing window | Statistical sentiment extremes — ±2σ flags BULLISH_EXTREME or BEARISH_EXTREME |
| `iv_percentile_zscore` | Per-ticker IV30d percentile + z-score over a trailing window | Outlier-robust replacement for IV-rank (Goyal-Saretto 2009) |
| `gex_time_series` | Multi-day Zero Gamma Level + total GEX trajectory | Detects dealer-hedging regime flips before realised vol expansion |
| `volatility_risk_premium` | IV30d − realised σ(30d) — VRP regime classifier | Premium-selling vs premium-buying environment (Bakshi-Kapadia 2003) |
| `signal_backtest` | Past signal → price outcome verification | "When this signal fired before, did the stock actually move?" |

### uw-watchlist — Watchlist Management
**When to use:** After you've identified your focus tickers. Cuts through 6K tickers to show only what you care about.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `manage_watchlist` | Add/remove/list tickers and groups | Build your watchlist (e.g., "earnings_plays", "momentum") |
| `watchlist_scan` | Comprehensive scan on watchlist only | Daily check on your tracked tickers |
| `watchlist_alerts` | Flag only unusual activity on watchlist | "Did anything notable happen on my tickers today?" |

### uw-playbook — Strategy Suggestions & Morning Briefings
**When to use:** You found an interesting ticker and want to know HOW to trade it. Also houses the daily synthesis morning briefing.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `suggest_strategy` | Multi-factor analysis → strategy suggestion | "What options strategy fits AAPL's current signals?" |
| `batch_strategy_scan` | Analyze multiple tickers for trade ideas | "Scan my watchlist — which tickers have a trade setup?" |
| `daily_synthesis` | Single-endpoint morning briefing (regime + confluence + watchlist) | Automated morning workflows — one call, structured JSON, LLM writes the prose |

### uw-risk — Risk & Market Regime
**When to use:** Before placing trades. Ensures you're not overconcentrated and trading with the macro trend.

| Tool | Description | When to Use |
|------|-------------|-------------|
| `portfolio_correlation` | Sector concentration and price correlation | "Are my 5 trade ideas actually 1 correlated bet?" |
| `market_regime` | SPY/VIX trend + breadth classification | "Should I be aggressive or defensive right now?" |

---

## Example Prompts

**Morning briefing (one call):**
> "Give me my morning briefing." *(uses `daily_synthesis` — returns regime, top confluence tickers, and watchlist alerts in one shot)*

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
