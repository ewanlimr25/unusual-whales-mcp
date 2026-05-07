"""MCP Server: Unusual Whales Dark Pool analysis."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

import pandas as pd

from shared.data_loader import load_data, list_available_dates
from shared.dp_classification import classify_dp_side
from shared.server_utils import text_result, df_to_result

server = Server("uw-darkpool")

# Tier ordering (ascending). Used for min_tier filtering.
_TIER_ORDER = ("retail", "large", "block", "mega")

# Lower bounds (inclusive) per tier. Retail is the catch-all below `large`.
_TIER_BOUNDARIES = {
    "mega": 10_000_000.0,
    "block": 1_000_000.0,
    "large": 100_000.0,
}


def _classify_tier(premium: float) -> str:
    """Classify a premium into a tier. Bounds are inclusive (>=)."""
    if premium >= _TIER_BOUNDARIES["mega"]:
        return "mega"
    if premium >= _TIER_BOUNDARIES["block"]:
        return "block"
    if premium >= _TIER_BOUNDARIES["large"]:
        return "large"
    return "retail"


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="largest_dark_pool_trades",
            description=(
                "Find the largest dark pool trades by premium or size. "
                "Shows NBBO context to indicate if traded at bid/ask/mid."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional)"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["premium", "size"],
                        "description": "Sort by premium or share size (default: premium)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="dark_pool_ticker_summary",
            description=(
                "Aggregate dark pool volume and premium per ticker, "
                "ranked by total premium. Shows institutional interest."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 30)", "default": 30},
                },
                "required": [],
            },
        ),
        Tool(
            name="dark_pool_price_levels",
            description=(
                "For a given ticker, show which price levels saw the most dark pool "
                "block activity. Reveals institutional support/resistance zones. "
                "Set days > 1 to aggregate across multiple recent sessions and find "
                "recurring institutional defense levels."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest. Ignored when days > 1."},
                    "symbol": {"type": "string", "description": "Ticker symbol (required)"},
                    "top_n": {"type": "integer", "description": "Number of price levels (default: 15)", "default": 15},
                    "days": {"type": "integer", "description": "Aggregate across this many recent sessions (default: 1 = single day)", "default": 1},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="extended_hours_filter",
            description=(
                "Isolate extended-hours dark pool trades. Pre/post-market "
                "block trades can signal next-day institutional moves."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="dp_block_size_stratified",
            description=(
                "Stratify dark pool premium into 4 tiers per ticker — mega (>=$10M), "
                "block (>=$1M), large (>=$100k), retail (<$100k) — with buy/sell ratio per "
                "tier. Conviction in mega/block tiers is the institutional smart-money "
                "signal; retail-tier moves are noise. Per Easley-Lopez de Prado-O'Hara 2012 "
                "(flow toxicity / informed-trading proxy)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "symbol": {"type": "string", "description": "Filter by ticker (optional)"},
                    "top_n": {"type": "integer", "description": "Number of tickers (default: 30)", "default": 30},
                    "min_tier": {
                        "type": "string",
                        "enum": ["retail", "large", "block", "mega"],
                        "description": "Only include tickers with at least one trade at this tier or higher (default: large)",
                        "default": "large",
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    date = arguments.get("date")
    top_n = arguments.get("top_n", 20)

    if name == "largest_dark_pool_trades":
        df = load_data("darkpool", date)
        if arguments.get("symbol"):
            df = df[df["ticker"] == arguments["symbol"].upper()]
        sort_col = arguments.get("sort_by", "premium")
        df["premium"] = df["premium"].astype(float)
        df = df.sort_values(sort_col, ascending=False).head(top_n)
        # Add bid/ask/mid context
        df["nbbo_mid"] = (df["nbbo_bid"].astype(float) + df["nbbo_ask"].astype(float)) / 2
        df["trade_vs_mid"] = df["price"].astype(float) - df["nbbo_mid"]
        cols = ["ticker", "executed_at", "price", "size", "premium",
                "nbbo_bid", "nbbo_ask", "nbbo_mid", "trade_vs_mid"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "dark_pool_ticker_summary":
        df = load_data("darkpool", date)
        df["premium"] = df["premium"].astype(float)
        df["size"] = df["size"].astype(float)
        agg = df.groupby("ticker").agg(
            total_premium=("premium", "sum"),
            total_shares=("size", "sum"),
            trade_count=("size", "count"),
            avg_price=("price", "mean"),
            max_single_trade=("premium", "max"),
        ).reset_index()
        agg = agg.sort_values("total_premium", ascending=False).head(top_n)
        return df_to_result(agg)

    elif name == "dark_pool_price_levels":
        symbol = arguments["symbol"].upper()
        days = max(1, int(arguments.get("days", 1)))
        if days == 1:
            df = load_data("darkpool", date)
            dates_covered = [date] if date else [list_available_dates("darkpool")[0]] if list_available_dates("darkpool") else []
        else:
            dates_to_load = list_available_dates("darkpool")[:days]
            if not dates_to_load:
                return text_result(f"No dark pool data available for {symbol}")
            frames = []
            for d in dates_to_load:
                try:
                    frames.append(load_data("darkpool", d))
                except Exception:
                    continue
            if not frames:
                return text_result(f"No dark pool data could be loaded for {symbol}")
            df = pd.concat(frames, ignore_index=True)
            dates_covered = dates_to_load
        df = df[df["ticker"] == symbol]
        if df.empty:
            return text_result({"symbol": symbol, "dates_covered": dates_covered, "results": [], "note": f"No dark pool trades found for {symbol}"})
        df["price"] = df["price"].astype(float)
        df["premium"] = df["premium"].astype(float)
        df["price_level"] = df["price"].round(2)
        agg = df.groupby("price_level").agg(
            total_premium=("premium", "sum"),
            total_shares=("size", "sum"),
            trade_count=("size", "count"),
        ).reset_index()
        agg = agg.sort_values("total_premium", ascending=False).head(top_n)
        return text_result({
            "symbol": symbol,
            "dates_covered": dates_covered,
            "results": agg.to_dict(orient="records"),
        })

    elif name == "dp_block_size_stratified":
        top_n = int(arguments.get("top_n", 30))
        min_tier = arguments.get("min_tier", "large")
        symbol = arguments.get("symbol")

        if top_n < 1 or top_n > 500:
            raise ValueError(f"top_n must be 1-500, got {top_n}")
        if min_tier not in _TIER_ORDER:
            raise ValueError(
                f"min_tier must be one of {_TIER_ORDER}, got {min_tier!r}"
            )

        try:
            df = load_data(
                "darkpool",
                date,
                usecols=["ticker", "price", "size", "premium", "nbbo_bid", "nbbo_ask"],
            )
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})

        if symbol:
            df = df[df["ticker"] == symbol.upper()]
        if df.empty:
            return text_result({
                "caveat": "Single-day dark pool stratification by premium tier.",
                "results": [],
            })

        df = df.copy()
        df["premium"] = pd.to_numeric(df["premium"], errors="coerce").fillna(0.0)
        df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0)
        df["tier"] = df["premium"].apply(_classify_tier)
        df = classify_dp_side(df)

        # Aggregate per (ticker, tier)
        df["_buy_size"] = df["size"].where(df["side"] == "buy", 0.0)
        df["_sell_size"] = df["size"].where(df["side"] == "sell", 0.0)
        grouped = (
            df.groupby(["ticker", "tier"], as_index=False)
            .agg(
                total_premium=("premium", "sum"),
                buy_volume=("_buy_size", "sum"),
                sell_volume=("_sell_size", "sum"),
                trade_count=("size", "count"),
            )
        )

        # Pivot into per-ticker dict-of-tiers
        min_tier_idx = _TIER_ORDER.index(min_tier)
        eligible_tiers = set(_TIER_ORDER[min_tier_idx:])

        results: list[dict] = []
        for ticker, group in grouped.groupby("ticker"):
            tier_data: dict[str, dict] = {
                t: {"total_premium": 0.0, "buy_volume": 0, "sell_volume": 0, "trade_count": 0, "buy_ratio": 0.5}
                for t in _TIER_ORDER
            }
            highest_tier_with_activity: str | None = None
            for _, row in group.iterrows():
                tier = row["tier"]
                buy_v = float(row["buy_volume"])
                sell_v = float(row["sell_volume"])
                total_v = buy_v + sell_v
                tier_data[tier] = {
                    "total_premium": round(float(row["total_premium"]), 2),
                    "buy_volume": int(buy_v),
                    "sell_volume": int(sell_v),
                    "trade_count": int(row["trade_count"]),
                    "buy_ratio": round(buy_v / total_v, 3) if total_v > 0 else 0.5,
                }
                if highest_tier_with_activity is None or _TIER_ORDER.index(tier) > _TIER_ORDER.index(highest_tier_with_activity):
                    highest_tier_with_activity = tier

            # Filter: include ticker only if it has activity at min_tier or higher
            if not any(tier_data[t]["trade_count"] > 0 for t in eligible_tiers):
                continue

            total_premium_all = sum(tier_data[t]["total_premium"] for t in _TIER_ORDER)
            results.append({
                "ticker": ticker,
                "total_premium_all_tiers": round(total_premium_all, 2),
                "highest_tier": highest_tier_with_activity,
                **tier_data,
            })

        # Rank by mega+block premium (institutional signal strength)
        results.sort(
            key=lambda r: r["mega"]["total_premium"] + r["block"]["total_premium"],
            reverse=True,
        )
        results = results[:top_n]

        return text_result({
            "caveat": (
                "Single-day stratification. Tier boundaries: mega (>=$10M), block (>=$1M), "
                "large (>=$100k), retail (<$100k). Conviction in mega/block tiers is the "
                "institutional smart-money signal; retail-tier moves are noise."
            ),
            "results": results,
        })

    elif name == "extended_hours_filter":
        df = load_data("darkpool", date)
        df = df[df["ext_hour_sold_codes"].notna() & (df["ext_hour_sold_codes"] != "")]
        if arguments.get("symbol"):
            df = df[df["ticker"] == arguments["symbol"].upper()]
        df["premium"] = df["premium"].astype(float)
        df = df.sort_values("premium", ascending=False).head(top_n)
        cols = ["ticker", "executed_at", "price", "size", "premium",
                "ext_hour_sold_codes", "nbbo_bid", "nbbo_ask"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
