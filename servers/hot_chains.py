"""MCP Server: Unusual Whales Hot Option Chains analysis."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import load_data
from shared.server_utils import text_result, df_to_result

server = Server("uw-hotchains")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="most_active_contracts",
            description=(
                "Find the most active option contracts by volume or premium. "
                "Shows the hottest contracts of the day."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "sort_by": {
                        "type": "string",
                        "enum": ["volume", "premium", "trades"],
                        "description": "Sort by volume, premium, or trade count (default: volume)",
                    },
                    "sector": {"type": "string", "description": "Filter by sector (optional)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="sweep_ratio_scanner",
            description=(
                "Find contracts with high sweep_volume/volume ratio \u2014 "
                "aggressive directional bets that sweep through multiple exchanges."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "min_volume": {"type": "integer", "description": "Min volume to filter noise (default: 500)", "default": 500},
                    "min_sweep_ratio": {"type": "number", "description": "Min sweep/volume ratio (default: 0.3)", "default": 0.3},
                },
                "required": [],
            },
        ),
        Tool(
            name="smart_money_flow",
            description=(
                "Compare ask_side_volume vs bid_side_volume per contract to gauge "
                "directional conviction. Ask-heavy = bullish, bid-heavy = bearish."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "direction": {
                        "type": "string",
                        "enum": ["bullish", "bearish"],
                        "description": "Show most bullish or bearish flow (default: bullish)",
                    },
                    "min_volume": {"type": "integer", "description": "Min volume (default: 500)", "default": 500},
                },
                "required": [],
            },
        ),
        Tool(
            name="multileg_activity",
            description=(
                "Highlight contracts with significant multileg and stock-multileg volume \u2014 "
                "indicates complex strategies (spreads, combos, hedges)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "min_multileg_ratio": {"type": "number", "description": "Min multileg/volume ratio (default: 0.3)", "default": 0.3},
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    date = arguments.get("date")
    top_n = arguments.get("top_n", 20)

    if name == "most_active_contracts":
        df = load_data("hotchains", date)
        if arguments.get("sector"):
            df = df[df["sector"] == arguments["sector"]]
        sort_map = {"volume": "volume", "premium": "premium", "trades": "trades"}
        sort_col = sort_map.get(arguments.get("sort_by", "volume"), "volume")
        df = df.sort_values(sort_col, ascending=False).head(top_n)
        cols = ["option_symbol", "date", "volume", "premium", "open_interest",
                "trades", "iv", "high", "low", "close", "sector",
                "ask_side_volume", "bid_side_volume", "sweep_volume"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "sweep_ratio_scanner":
        df = load_data("hotchains", date)
        min_vol = arguments.get("min_volume", 500)
        min_ratio = arguments.get("min_sweep_ratio", 0.3)
        df = df[df["volume"] >= min_vol]
        df["sweep_ratio"] = df["sweep_volume"] / df["volume"]
        df = df[df["sweep_ratio"] >= min_ratio]
        df = df.sort_values("sweep_ratio", ascending=False).head(top_n)
        cols = ["option_symbol", "volume", "sweep_volume", "sweep_ratio",
                "premium", "open_interest", "iv", "close", "sector"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "smart_money_flow":
        df = load_data("hotchains", date)
        min_vol = arguments.get("min_volume", 500)
        df = df[df["volume"] >= min_vol]
        df["ask_bid_ratio"] = df["ask_side_volume"] / df["bid_side_volume"].replace(0, 1)
        df["net_flow"] = df["ask_side_volume"] - df["bid_side_volume"]
        direction = arguments.get("direction", "bullish")
        ascending = direction == "bearish"
        df = df.sort_values("net_flow", ascending=ascending).head(top_n)
        cols = ["option_symbol", "volume", "ask_side_volume", "bid_side_volume",
                "net_flow", "ask_bid_ratio", "premium", "iv", "close", "sector"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "multileg_activity":
        df = load_data("hotchains", date)
        df = df[df["volume"] > 0]
        df["multileg_total"] = df["multileg_volume"].fillna(0) + df["stock_multi_leg_volume"].fillna(0)
        df["multileg_ratio"] = df["multileg_total"] / df["volume"]
        min_ratio = arguments.get("min_multileg_ratio", 0.3)
        df = df[df["multileg_ratio"] >= min_ratio]
        df = df.sort_values("multileg_total", ascending=False).head(top_n)
        cols = ["option_symbol", "volume", "multileg_volume", "stock_multi_leg_volume",
                "multileg_total", "multileg_ratio", "premium", "iv", "sector"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
