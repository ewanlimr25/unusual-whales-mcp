"""MCP Server: trade-level options flow analysis.

Phase 2 split: holds the trade-level flow tools (sweeps, premium, volume/OI,
sector flow). Per-ticker structural tools (Greeks, IV term structure, GEX, DEX)
moved to `uw-options-structure`.

Tools:
  - top_premium_trades        Largest single options trades by premium
  - unusual_volume_scanner    Per-contract Vol/OI ratio outliers
  - sweep_detector            Aggressive sweeps by side, contract-aggregated
  - iv_outliers               Per-contract high-IV contracts
  - greek_screener            Filter trades by Greeks
  - expiry_heatmap            Volume/premium concentration by expiry
  - sector_flow_summary       Premium by sector × call/put
  - sector_flow_persistence   Multi-day sector net-flow consistency
  - dte_volume_share          0DTE / weekly / monthly / LEAP volume share
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date as date_cls
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import list_available_dates, load_data
from shared.server_utils import df_to_result, text_result

server = Server("uw-options-flow")

PRICE_COLS = [
    "executed_at", "underlying_symbol", "strike", "option_type", "expiry",
    "underlying_price", "price", "size", "premium", "volume", "open_interest",
    "implied_volatility", "delta", "theta", "gamma", "vega", "side", "sector",
]


def _load_options(date=None, usecols=None, nrows=None):
    kwargs = {}
    if usecols:
        kwargs["usecols"] = usecols
    if nrows:
        kwargs["nrows"] = nrows
    return load_data("options", date, **kwargs)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="top_premium_trades",
            description=(
                "Find the largest premium options trades of the day. "
                "Shows the biggest single-trade bets by dollar premium."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional)"},
                    "option_type": {"type": "string", "enum": ["call", "put"], "description": "Filter by call/put (optional)"},
                    "sector": {"type": "string", "description": "Filter by sector (optional)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="unusual_volume_scanner",
            description=(
                "Find options contracts where trade volume far exceeds open interest — "
                "signals new large positions being opened."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional)"},
                    "min_volume": {"type": "integer", "description": "Min volume threshold (default: 100)", "default": 100},
                    "min_vol_oi_ratio": {"type": "number", "description": "Min volume/OI ratio (default: 5.0)", "default": 5.0},
                },
                "required": [],
            },
        ),
        Tool(
            name="sweep_detector",
            description=(
                "Identify aggressive sweeps — trades hitting the ask (bullish) or bid (bearish). "
                "Aggregates by contract to show conviction."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional)"},
                    "side": {"type": "string", "enum": ["ask", "bid"], "description": "Filter by trade side (optional)"},
                    "min_premium": {"type": "number", "description": "Min total premium (default: 100000)", "default": 100000},
                },
                "required": [],
            },
        ),
        Tool(
            name="iv_outliers",
            description=(
                "Find options with unusually high implied volatility — "
                "may indicate expected big moves or mispricing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional)"},
                    "min_iv": {"type": "number", "description": "Min IV threshold (default: 1.0 = 100%)", "default": 1.0},
                    "min_volume": {"type": "integer", "description": "Min volume to filter illiquid (default: 50)", "default": 50},
                },
                "required": [],
            },
        ),
        Tool(
            name="sector_flow_summary",
            description=(
                "Aggregate premium by sector and call/put to show where money is flowing. "
                "Reveals sector-level bullish/bearish sentiment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                },
                "required": [],
            },
        ),
        Tool(
            name="expiry_heatmap",
            description=(
                "Aggregate premium and volume by expiry date to see where "
                "positioning is concentrated (weeklies vs monthlies vs LEAPS)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional)"},
                    "top_n": {"type": "integer", "description": "Number of expiry dates to show (default: 20)", "default": 20},
                },
                "required": [],
            },
        ),
        Tool(
            name="greek_screener",
            description=(
                "Screen trades by Greeks — find high-delta directional bets, "
                "high-gamma near-the-money plays, or high-vega volatility bets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of results (default: 20)", "default": 20},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional)"},
                    "min_delta": {"type": "number", "description": "Min absolute delta (optional)"},
                    "max_delta": {"type": "number", "description": "Max absolute delta (optional)"},
                    "min_gamma": {"type": "number", "description": "Min gamma (optional)"},
                    "min_vega": {"type": "number", "description": "Min vega (optional)"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["premium", "delta", "gamma", "vega"],
                        "description": "Sort results by (default: premium)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="dte_volume_share",
            description=(
                "Share of total option volume by DTE bucket — 0DTE / weeklies / monthlies / LEAPs. "
                "High 0DTE share = retail-driven session; high monthly+LEAP share = institutional. "
                "Regime hint helps weight other signals."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Filter by ticker (optional; defaults to whole market)"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                },
                "required": [],
            },
        ),
        Tool(
            name="sector_flow_persistence",
            description=(
                "Multi-day version of sector_flow_summary. Returns per-sector net flow across "
                "the most recent N sessions and a persistence score (fraction of days flow held "
                "the same sign). Used to detect durable sector rotations vs single-day noise."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "Number of recent sessions (default: 5)", "default": 5},
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    date = arguments.get("date")
    top_n = arguments.get("top_n", 20)

    if name == "top_premium_trades":
        try:
            df = _load_options(date)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        if arguments.get("option_type"):
            df = df[df["option_type"] == arguments["option_type"]]
        if arguments.get("sector"):
            df = df[df["sector"] == arguments["sector"]]
        df = df.sort_values("premium", ascending=False).head(top_n)
        cols = ["executed_at", "underlying_symbol", "option_type", "strike", "expiry",
                "side", "price", "size", "premium", "underlying_price",
                "implied_volatility", "delta", "sector"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "unusual_volume_scanner":
        try:
            df = _load_options(date)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        min_vol = arguments.get("min_volume", 100)
        min_ratio = arguments.get("min_vol_oi_ratio", 5.0)
        agg = df.groupby(["underlying_symbol", "option_type", "strike", "expiry"]).agg(
            total_volume=("size", "sum"),
            total_premium=("premium", "sum"),
            open_interest=("open_interest", "max"),
            avg_iv=("implied_volatility", "mean"),
            trade_count=("size", "count"),
        ).reset_index()
        agg = agg[agg["total_volume"] >= min_vol]
        agg = agg[agg["open_interest"] > 0]
        agg["vol_oi_ratio"] = agg["total_volume"] / agg["open_interest"]
        agg = agg[agg["vol_oi_ratio"] >= min_ratio]
        agg = agg.sort_values("vol_oi_ratio", ascending=False).head(top_n)
        return df_to_result(agg)

    elif name == "sweep_detector":
        try:
            df = _load_options(date)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        if arguments.get("side"):
            df = df[df["side"] == arguments["side"]]
        agg = df.groupby(["underlying_symbol", "option_type", "strike", "expiry", "side"]).agg(
            total_premium=("premium", "sum"),
            total_size=("size", "sum"),
            trade_count=("size", "count"),
            avg_price=("price", "mean"),
        ).reset_index()
        min_prem = arguments.get("min_premium", 100000)
        agg = agg[agg["total_premium"] >= min_prem]
        agg = agg.sort_values("total_premium", ascending=False).head(top_n)
        return df_to_result(agg)

    elif name == "iv_outliers":
        try:
            df = _load_options(date)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        min_iv = arguments.get("min_iv", 1.0)
        min_vol = arguments.get("min_volume", 50)
        df = df[df["implied_volatility"] >= min_iv]
        agg = df.groupby(["underlying_symbol", "option_type", "strike", "expiry"]).agg(
            max_iv=("implied_volatility", "max"),
            avg_iv=("implied_volatility", "mean"),
            total_volume=("size", "sum"),
            total_premium=("premium", "sum"),
        ).reset_index()
        agg = agg[agg["total_volume"] >= min_vol]
        agg = agg.sort_values("max_iv", ascending=False).head(top_n)
        return df_to_result(agg)

    elif name == "sector_flow_summary":
        try:
            df = _load_options(date)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        df = df[df["sector"].notna() & (df["sector"] != "")]
        agg = df.groupby(["sector", "option_type"]).agg(
            total_premium=("premium", "sum"),
            total_volume=("size", "sum"),
            trade_count=("size", "count"),
            avg_iv=("implied_volatility", "mean"),
        ).reset_index()
        pivot = agg.pivot_table(
            index="sector",
            columns="option_type",
            values=["total_premium", "total_volume"],
            fill_value=0,
        )
        pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
        pivot = pivot.reset_index()
        if "total_premium_call" in pivot.columns and "total_premium_put" in pivot.columns:
            pivot["net_flow"] = pivot["total_premium_call"] - pivot["total_premium_put"]
            pivot = pivot.sort_values("net_flow", ascending=False)
        return df_to_result(pivot, max_rows=30)

    elif name == "expiry_heatmap":
        try:
            df = _load_options(date)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        agg = df.groupby(["expiry", "option_type"]).agg(
            total_premium=("premium", "sum"),
            total_volume=("size", "sum"),
            trade_count=("size", "count"),
        ).reset_index()
        pivot = agg.pivot_table(
            index="expiry",
            columns="option_type",
            values=["total_premium", "total_volume"],
            fill_value=0,
        )
        pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
        pivot = pivot.reset_index()
        pivot["total"] = pivot.get("total_premium_call", 0) + pivot.get("total_premium_put", 0)
        pivot = pivot.sort_values("total", ascending=False).head(top_n)
        return df_to_result(pivot)

    elif name == "greek_screener":
        try:
            df = _load_options(date)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        df = df.copy()
        df["abs_delta"] = df["delta"].abs()
        if arguments.get("min_delta") is not None:
            df = df[df["abs_delta"] >= arguments["min_delta"]]
        if arguments.get("max_delta") is not None:
            df = df[df["abs_delta"] <= arguments["max_delta"]]
        if arguments.get("min_gamma") is not None:
            df = df[df["gamma"] >= arguments["min_gamma"]]
        if arguments.get("min_vega") is not None:
            df = df[df["vega"] >= arguments["min_vega"]]
        sort_col = arguments.get("sort_by", "premium")
        if sort_col == "delta":
            sort_col = "abs_delta"
        df = df.sort_values(sort_col, ascending=False).head(top_n)
        cols = ["executed_at", "underlying_symbol", "option_type", "strike", "expiry",
                "side", "premium", "size", "delta", "gamma", "vega", "theta",
                "implied_volatility", "underlying_price"]
        return df_to_result(df[[c for c in cols if c in df.columns]])

    elif name == "dte_volume_share":
        cols = ["underlying_symbol", "expiry", "size"]
        try:
            df = _load_options(date, usecols=cols)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        if df.empty:
            return text_result({"note": "no data"})
        df = df.copy()
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df = df.dropna(subset=["expiry"])
        today = date_cls.today()
        df["dte"] = (df["expiry"].dt.date - today).apply(lambda d: d.days)
        total = float(df["size"].sum())
        if total == 0:
            return text_result({"note": "zero volume"})
        share_0dte = float(df[df["dte"] == 0]["size"].sum()) / total
        share_weeklies = float(df[(df["dte"] >= 1) & (df["dte"] <= 7)]["size"].sum()) / total
        share_monthlies = float(df[(df["dte"] >= 8) & (df["dte"] <= 45)]["size"].sum()) / total
        share_leaps = float(df[df["dte"] > 180]["size"].sum()) / total
        if share_0dte > 0.4:
            hint = "RETAIL_DRIVEN"
        elif (share_monthlies + share_leaps) > 0.6:
            hint = "INSTITUTIONAL"
        else:
            hint = "BALANCED"
        return text_result({
            "symbol": arguments.get("symbol", "MARKET"),
            "share_0dte": round(share_0dte, 3),
            "share_weeklies": round(share_weeklies, 3),
            "share_monthlies": round(share_monthlies, 3),
            "share_leaps": round(share_leaps, 3),
            "regime_hint": hint,
        })

    elif name == "sector_flow_persistence":
        days = int(arguments.get("days", 5))
        dates = list_available_dates("options")[:days]
        if not dates:
            return text_result({"note": "no options data available"})
        per_day = {}
        all_sectors = set()
        for d in dates:
            try:
                df = _load_options(d, usecols=["sector", "option_type", "premium"])
                df = df[df["sector"].notna() & (df["sector"] != "")]
                pivot = df.pivot_table(index="sector", columns="option_type", values="premium", aggfunc="sum", fill_value=0)
                for sector, row in pivot.iterrows():
                    call_p = float(row.get("call", 0))
                    put_p = float(row.get("put", 0))
                    per_day.setdefault(sector, {})[d] = round(call_p - put_p, 2)
                    all_sectors.add(sector)
            except Exception:
                continue
        results = []
        for sector in all_sectors:
            day_flows = per_day.get(sector, {})
            if not day_flows:
                continue
            signs = [1 if v > 0 else -1 if v < 0 else 0 for v in day_flows.values()]
            if not signs:
                continue
            dominant = 1 if sum(s for s in signs if s > 0) > abs(sum(s for s in signs if s < 0)) else -1
            persistence = sum(1 for s in signs if s == dominant) / len(signs)
            if persistence >= 0.8:
                trend = "INFLOW" if dominant > 0 else "OUTFLOW"
            else:
                trend = "ROTATING"
            results.append({
                "sector": sector,
                "net_flow_by_day": day_flows,
                "persistence_score": round(persistence, 3),
                "trend": trend,
            })
        results.sort(key=lambda r: r["persistence_score"], reverse=True)
        return text_result({
            "days": days,
            "dates_covered": dates,
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
