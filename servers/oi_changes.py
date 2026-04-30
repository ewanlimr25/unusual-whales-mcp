"""MCP Server: Unusual Whales Open Interest Changes analysis."""

import asyncio
import sys
from pathlib import Path

import pandas as pd

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
        Tool(
            name="position_rolling_detector",
            description=(
                "Detect same-day position rolls: tickers where near-term OI decreased "
                "AND far-term OI increased on the same day, inferring an institution closed "
                "a near-expiry position and re-opened further out. "
                "Caveat: single-day only — cross-session rolls are not detected."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "symbol": {"type": "string", "description": "Filter by underlying ticker (optional)"},
                    "threshold": {
                        "type": "integer",
                        "description": "Min absolute OI change in each bucket to signal a roll (default: 1000)",
                        "default": 1000,
                    },
                    "near_dte_max": {
                        "type": "integer",
                        "description": "Max DTE to classify as near-term bucket (default: 21)",
                        "default": 21,
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

    elif name == "position_rolling_detector":
        df = load_data("oi", date)
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        threshold = arguments.get("threshold", 1000)
        near_dte_max = arguments.get("near_dte_max", 21)

        df = df.copy()
        df["oi_diff_plain"] = pd.to_numeric(df["oi_diff_plain"], errors="coerce")
        df["dte"] = pd.to_numeric(df["dte"], errors="coerce")
        df = df.dropna(subset=["oi_diff_plain", "dte"])

        # Infer option_type from option_symbol (OCC symbology: C=call, P=put)
        df["is_call"] = df["option_symbol"].str.contains("C", na=False)
        df["option_type_inferred"] = df["is_call"].map({True: "call", False: "put"})
        df["bucket"] = df["dte"].apply(lambda d: "near" if d <= near_dte_max else "far")

        grp = (
            df.groupby(["underlying_symbol", "option_type_inferred", "bucket"])["oi_diff_plain"]
            .sum()
            .reset_index()
        )

        pivot = grp.pivot_table(
            index=["underlying_symbol", "option_type_inferred"],
            columns="bucket",
            values="oi_diff_plain",
            fill_value=0,
        ).reset_index()
        pivot.columns.name = None

        for col in ["near", "far"]:
            if col not in pivot.columns:
                pivot[col] = 0.0

        rolls = pivot[(pivot["near"] < -threshold) & (pivot["far"] > threshold)].copy()
        rolls["roll_size"] = rolls.apply(lambda r: min(abs(r["near"]), r["far"]), axis=1)
        rolls["balance_ratio"] = rolls.apply(
            lambda r: round(min(abs(r["near"]), r["far"]) / max(abs(r["near"]), r["far"]), 3),
            axis=1,
        )
        rolls = rolls.sort_values("roll_size", ascending=False)

        results = [
            {
                "symbol": row["underlying_symbol"],
                "option_type": row["option_type_inferred"],
                "near_oi_change": int(row["near"]),
                "far_oi_change": int(row["far"]),
                "roll_size": int(row["roll_size"]),
                "balance_ratio": float(row["balance_ratio"]),
            }
            for _, row in rolls.iterrows()
        ]

        return text_result({
            "caveat": (
                "Single-day detection only — cross-session rolls are not captured. "
                "Detected rolls reflect contracts where near-DTE OI decreased "
                "and far-DTE OI increased on the same trading day."
            ),
            "threshold": threshold,
            "near_dte_max": near_dte_max,
            "rolls_detected": len(results),
            "results": results,
        })

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
