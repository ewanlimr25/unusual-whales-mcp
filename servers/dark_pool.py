"""MCP Server: Unusual Whales Dark Pool analysis."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import load_data, list_available_dates
from shared.server_utils import text_result, df_to_result

server = Server("uw-darkpool")


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
                "block activity. Reveals institutional support/resistance zones."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "symbol": {"type": "string", "description": "Ticker symbol (required)"},
                    "top_n": {"type": "integer", "description": "Number of price levels (default: 15)", "default": 15},
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
        df = load_data("darkpool", date)
        symbol = arguments["symbol"].upper()
        df = df[df["ticker"] == symbol]
        if df.empty:
            return text_result(f"No dark pool trades found for {symbol}")
        df["price"] = df["price"].astype(float)
        df["premium"] = df["premium"].astype(float)
        # Round price to nearest cent for grouping
        df["price_level"] = df["price"].round(2)
        agg = df.groupby("price_level").agg(
            total_premium=("premium", "sum"),
            total_shares=("size", "sum"),
            trade_count=("size", "count"),
        ).reset_index()
        agg = agg.sort_values("total_premium", ascending=False).head(top_n)
        return df_to_result(agg)

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
