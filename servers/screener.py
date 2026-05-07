"""MCP Server: Unusual Whales Stock Screener analysis."""

import asyncio
import sys
from pathlib import Path

# Allow running as module from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import load_data, list_available_dates
from shared.server_utils import text_result, df_to_result

server = Server("uw-screener")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="bullish_bearish_screener",
            description=(
                "Rank tickers by net bullish vs bearish premium flow. "
                "Shows which stocks have the most aggressive directional bets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "direction": {
                        "type": "string",
                        "enum": ["bullish", "bearish"],
                        "description": "Filter for most bullish or most bearish (default: bullish)",
                    },
                    "sector": {"type": "string", "description": "Filter by sector (optional)"},
                    "min_premium": {"type": "number", "description": "Min total premium to filter noise (optional)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="iv_rank_screener",
            description=(
                "Find tickers with extreme IV rank. High IV rank = expensive options "
                "(good for selling premium). Low IV rank = cheap options (good for buying)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "mode": {
                        "type": "string",
                        "enum": ["high", "low"],
                        "description": "Show highest or lowest IV rank (default: high)",
                    },
                    "min_marketcap": {"type": "number", "description": "Min market cap filter (optional)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="put_call_ratio_extremes",
            description=(
                "[DEPRECATED v0.4.0] Use `pc_ratio_zscore` from uw-historical instead \u2014 "
                "it computes a proper statistical z-score over a trailing window which "
                "is robust to single-day outliers. This tool returns raw P/C extremes "
                "for one day only (high P/C may indicate hedging or bearish bets, low "
                "P/C indicates bullish activity). Will be removed in v0.5.0."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "mode": {
                        "type": "string",
                        "enum": ["high", "low"],
                        "description": "Show highest or lowest P/C ratio (default: high)",
                    },
                    "min_total_volume": {"type": "integer", "description": "Min options volume to filter noise (optional)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="earnings_catalyst_scanner",
            description=(
                "Find tickers with upcoming earnings + elevated IV. "
                "Great for identifying potential earnings plays."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "days_until_earnings": {"type": "integer", "description": "Max days until earnings (default: 14)", "default": 14},
                    "min_iv_rank": {"type": "number", "description": "Min IV rank (default: 50)", "default": 50},
                },
                "required": [],
            },
        ),
        Tool(
            name="volume_vs_average",
            description=(
                "Find tickers where today's options volume significantly exceeds "
                "their 30-day average \u2014 signals unusual activity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "min_volume_ratio": {"type": "number", "description": "Min ratio vs 30-day avg (default: 2.0)", "default": 2.0},
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    date = arguments.get("date")
    top_n = arguments.get("top_n", 20)

    if name == "bullish_bearish_screener":
        df = load_data("screener", date)
        direction = arguments.get("direction", "bullish")
        df["net_flow"] = df["bullish_premium"].fillna(0) - df["bearish_premium"].fillna(0)
        if arguments.get("sector"):
            df = df[df["sector"] == arguments["sector"]]
        if arguments.get("min_premium"):
            total = df["call_premium"].fillna(0) + df["put_premium"].fillna(0)
            df = df[total >= arguments["min_premium"]]
        ascending = direction == "bearish"
        df = df.sort_values("net_flow", ascending=ascending).head(top_n)
        cols = ["ticker", "full_name", "sector", "close", "bullish_premium", "bearish_premium",
                "net_flow", "call_volume", "put_volume", "put_call_ratio", "iv_rank"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "iv_rank_screener":
        df = load_data("screener", date)
        df["iv_rank"] = df["iv_rank"].astype(float)
        if arguments.get("min_marketcap"):
            df = df[df["marketcap"].astype(float) >= arguments["min_marketcap"]]
        mode = arguments.get("mode", "high")
        ascending = mode == "low"
        df = df.sort_values("iv_rank", ascending=ascending).head(top_n)
        cols = ["ticker", "full_name", "sector", "close", "iv_rank", "volatility",
                "iv30d", "iv30d_1w", "iv30d_1m", "put_call_ratio", "marketcap"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "put_call_ratio_extremes":
        df = load_data("screener", date)
        if arguments.get("min_total_volume"):
            df = df[(df["call_volume"] + df["put_volume"]) >= arguments["min_total_volume"]]
        mode = arguments.get("mode", "high")
        ascending = mode == "low"
        df = df.sort_values("put_call_ratio", ascending=ascending).head(top_n)
        cols = ["ticker", "full_name", "sector", "close", "put_call_ratio",
                "call_volume", "put_volume", "call_premium", "put_premium", "iv_rank"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "earnings_catalyst_scanner":
        import pandas as pd
        df = load_data("screener", date)
        df["next_earnings_date"] = pd.to_datetime(df["next_earnings_date"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df["days_to_earnings"] = (df["next_earnings_date"] - df["date"]).dt.days
        max_days = arguments.get("days_until_earnings", 14)
        min_iv = arguments.get("min_iv_rank", 50)
        df = df[(df["days_to_earnings"] > 0) & (df["days_to_earnings"] <= max_days)]
        df = df[df["iv_rank"].astype(float) >= min_iv]
        df = df.sort_values("iv_rank", ascending=False)
        cols = ["ticker", "full_name", "sector", "close", "next_earnings_date", "er_time",
                "days_to_earnings", "iv_rank", "implied_move", "implied_move_perc",
                "call_volume", "put_volume", "put_call_ratio"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "volume_vs_average":
        df = load_data("screener", date)
        df["total_volume"] = df["call_volume"].fillna(0) + df["put_volume"].fillna(0)
        df["avg_total_volume"] = df["avg_30_day_call_volume"].fillna(0) + df["avg_30_day_put_volume"].fillna(0)
        df = df[df["avg_total_volume"] > 0]
        df["volume_ratio"] = df["total_volume"] / df["avg_total_volume"]
        min_ratio = arguments.get("min_volume_ratio", 2.0)
        df = df[df["volume_ratio"] >= min_ratio]
        df = df.sort_values("volume_ratio", ascending=False).head(top_n)
        cols = ["ticker", "full_name", "sector", "close", "total_volume",
                "avg_total_volume", "volume_ratio", "put_call_ratio", "iv_rank"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
