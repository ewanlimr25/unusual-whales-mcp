"""MCP Server: Unusual Whales Hot Option Chains analysis."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import load_data, list_available_dates
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
            name="multi_day_sweep_persistence",
            description=(
                "Find tickers that appear in top sweep activity across multiple recent sessions. "
                "Single-day sweeps are news; multi-day sweep campaigns are conviction. Returns "
                "consistency_score = sessions_in_top / days, plus dominant_direction."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Number of recent sessions to scan (default: 5)", "default": 5},
                    "top_n": {"type": "integer", "description": "Top-N sweep tickers to track per session (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter to a specific ticker (optional)"},
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

    elif name == "multi_day_sweep_persistence":
        days = int(arguments.get("days", 5))
        per_session_top = int(arguments.get("top_n", 20))
        symbol_filter = arguments.get("symbol")
        if symbol_filter:
            symbol_filter = symbol_filter.upper()
        dates = list_available_dates("hotchains")[:days]
        if not dates:
            return text_result({"note": "no hot chains data"})

        per_ticker_sessions = {}
        per_ticker_premium = {}
        per_ticker_directions = {}
        for d in dates:
            try:
                df = load_data("hotchains", d)
                df = df[df["volume"] > 0].copy()
                df["sweep_volume"] = df["sweep_volume"].fillna(0)
                df = df[df["sweep_volume"] > 0]
                df["ticker"] = df["option_symbol"].str.extract(r"^([A-Z]+)")[0]
                if symbol_filter:
                    df = df[df["ticker"] == symbol_filter]
                df["ask_side_volume"] = df["ask_side_volume"].fillna(0)
                df["bid_side_volume"] = df["bid_side_volume"].fillna(0)
                ticker_agg = df.groupby("ticker").agg(
                    total_sweep_premium=("premium", "sum"),
                    total_ask=("ask_side_volume", "sum"),
                    total_bid=("bid_side_volume", "sum"),
                ).reset_index()
                ticker_agg = ticker_agg.sort_values("total_sweep_premium", ascending=False).head(per_session_top)
                for _, row in ticker_agg.iterrows():
                    t = row["ticker"]
                    if not t:
                        continue
                    per_ticker_sessions[t] = per_ticker_sessions.get(t, 0) + 1
                    per_ticker_premium[t] = per_ticker_premium.get(t, 0.0) + float(row["total_sweep_premium"])
                    direction = "bullish" if row["total_ask"] > row["total_bid"] else "bearish"
                    per_ticker_directions.setdefault(t, []).append(direction)
            except Exception:
                continue

        results = []
        for t, sessions in per_ticker_sessions.items():
            dirs = per_ticker_directions.get(t, [])
            bull = sum(1 for d in dirs if d == "bullish")
            bear = len(dirs) - bull
            if bull > bear * 1.5:
                dominant = "bullish"
            elif bear > bull * 1.5:
                dominant = "bearish"
            else:
                dominant = "mixed"
            results.append({
                "ticker": t,
                "sessions_in_top": sessions,
                "total_sweep_premium": round(per_ticker_premium[t], 2),
                "dominant_direction": dominant,
                "consistency_score": round(sessions / len(dates), 3),
            })
        results.sort(key=lambda r: (r["consistency_score"], r["total_sweep_premium"]), reverse=True)
        return text_result({
            "days": len(dates),
            "dates_covered": dates,
            "results": results,
        })

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
