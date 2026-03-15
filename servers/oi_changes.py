"""MCP Server: Unusual Whales Open Interest Changes analysis."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import load_data
from shared.server_utils import text_result, df_to_result

server = Server("uw-oi")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="biggest_oi_increases",
            description=(
                "Find contracts with the largest open interest increases \u2014 "
                "new money flowing into positions. Signals fresh conviction."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by underlying ticker (optional)"},
                    "min_oi_change": {"type": "integer", "description": "Min absolute OI change (optional)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="oi_decrease_with_volume",
            description=(
                "Find contracts where OI decreased alongside high volume \u2014 "
                "positions being closed. May signal profit-taking or stop-outs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by underlying ticker (optional)"},
                    "min_volume": {"type": "integer", "description": "Min volume (default: 100)", "default": 100},
                },
                "required": [],
            },
        ),
        Tool(
            name="smart_positioning",
            description=(
                "Cross-reference OI changes with ask/bid side volume to determine "
                "if new positions are bullish (ask-side opening) or bearish (bid-side opening)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by underlying ticker (optional)"},
                    "direction": {
                        "type": "string",
                        "enum": ["bullish", "bearish"],
                        "description": "Filter by inferred direction (optional)",
                    },
                    "min_oi_change": {"type": "integer", "description": "Min OI increase (default: 500)", "default": 500},
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    date = arguments.get("date")
    top_n = arguments.get("top_n", 20)

    if name == "biggest_oi_increases":
        df = load_data("oi", date)
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        df["oi_diff_plain"] = df["oi_diff_plain"].astype(float)
        if arguments.get("min_oi_change"):
            df = df[df["oi_diff_plain"] >= arguments["min_oi_change"]]
        df = df[df["oi_diff_plain"] > 0]
        df = df.sort_values("oi_diff_plain", ascending=False).head(top_n)
        cols = ["option_symbol", "underlying_symbol", "strike", "last_oi", "curr_oi",
                "oi_diff_plain", "oi_change", "volume", "avg_price", "prev_total_premium",
                "prev_ask_volume", "prev_bid_volume", "sector", "stock_price", "dte"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "oi_decrease_with_volume":
        df = load_data("oi", date)
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        df["oi_diff_plain"] = df["oi_diff_plain"].astype(float)
        min_vol = arguments.get("min_volume", 100)
        df = df[(df["oi_diff_plain"] < 0) & (df["volume"] >= min_vol)]
        df = df.sort_values("oi_diff_plain", ascending=True).head(top_n)
        cols = ["option_symbol", "underlying_symbol", "strike", "last_oi", "curr_oi",
                "oi_diff_plain", "volume", "avg_price", "prev_total_premium",
                "sector", "stock_price", "dte"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "smart_positioning":
        df = load_data("oi", date)
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        df["oi_diff_plain"] = df["oi_diff_plain"].astype(float)
        min_oi = arguments.get("min_oi_change", 500)
        # Only look at OI increases (new positions)
        df = df[df["oi_diff_plain"] >= min_oi]
        # Determine direction: ask-heavy = buying (bullish for calls, bearish for puts)
        df["prev_ask_volume"] = df["prev_ask_volume"].astype(float)
        df["prev_bid_volume"] = df["prev_bid_volume"].astype(float)
        df["net_ask_bid"] = df["prev_ask_volume"] - df["prev_bid_volume"]
        # Infer if this is a call or put from the option symbol
        df["is_call"] = df["option_symbol"].str.contains("C")
        # Bullish = call bought on ask OR put sold on bid
        # Bearish = put bought on ask OR call sold on bid
        df["inferred_direction"] = "neutral"
        df.loc[(df["is_call"]) & (df["net_ask_bid"] > 0), "inferred_direction"] = "bullish"
        df.loc[(df["is_call"]) & (df["net_ask_bid"] < 0), "inferred_direction"] = "bearish"
        df.loc[(~df["is_call"]) & (df["net_ask_bid"] > 0), "inferred_direction"] = "bearish"
        df.loc[(~df["is_call"]) & (df["net_ask_bid"] < 0), "inferred_direction"] = "bullish"

        if arguments.get("direction"):
            df = df[df["inferred_direction"] == arguments["direction"]]

        df = df.sort_values("oi_diff_plain", ascending=False).head(top_n)
        cols = ["option_symbol", "underlying_symbol", "strike", "oi_diff_plain",
                "prev_ask_volume", "prev_bid_volume", "net_ask_bid",
                "inferred_direction", "prev_total_premium", "stock_price", "dte", "sector"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
