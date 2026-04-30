"""MCP Server: Unusual Whales All Options flow analysis."""

import asyncio
import sys
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import load_data, list_available_dates
from shared.server_utils import text_result, df_to_result

server = Server("uw-options")

# Column subsets for efficient loading
PRICE_COLS = [
    "executed_at", "underlying_symbol", "strike", "option_type", "expiry",
    "underlying_price", "price", "size", "premium", "volume", "open_interest",
    "implied_volatility", "delta", "theta", "gamma", "vega", "side", "sector",
]


def _load_options(date=None, usecols=None, nrows=None):
    """Load options data with optional column/row limits for the 9.8M row file."""
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
                "Find options contracts where trade volume far exceeds open interest \u2014 "
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
                "Identify aggressive sweeps \u2014 trades hitting the ask (bullish) or bid (bearish). "
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
                "Find options with unusually high implied volatility \u2014 "
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
                "Screen trades by Greeks \u2014 find high-delta directional bets, "
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
            name="iv_term_structure",
            description=(
                "Aggregate average implied volatility by expiry to identify the term structure shape. "
                "BACKWARDATION (front IV > back IV) signals an imminent event or panic buying. "
                "CONTANGO is the normal upward slope. KINKED indicates a binary event at a specific expiry."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "symbol": {"type": "string", "description": "Filter by ticker symbol (optional but recommended)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="gamma_exposure_profile",
            description=(
                "Calculate net dealer Gamma Exposure (GEX) per strike for a symbol. "
                "Positive GEX = dealers long gamma, price-stabilizing (pinning). "
                "Negative GEX = dealers short gamma, trend-amplifying. "
                "Returns the Zero Gamma Level where dealer hedging pressure flips."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (required \u2014 GEX is per-underlying)"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "include_zero_gamma": {
                        "type": "boolean",
                        "description": "Compute zero gamma flip level (default: true)",
                        "default": True,
                    },
                },
                "required": ["symbol"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    date = arguments.get("date")
    top_n = arguments.get("top_n", 20)

    if name == "top_premium_trades":
        df = _load_options(date)
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
        df = _load_options(date)
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        min_vol = arguments.get("min_volume", 100)
        min_ratio = arguments.get("min_vol_oi_ratio", 5.0)
        # Aggregate by contract
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
        df = _load_options(date)
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        if arguments.get("side"):
            df = df[df["side"] == arguments["side"]]
        # Aggregate by contract + side
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
        df = _load_options(date)
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
        min_iv = arguments.get("min_iv", 1.0)
        min_vol = arguments.get("min_volume", 50)
        df = df[df["implied_volatility"] >= min_iv]
        # Aggregate by contract
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
        df = _load_options(date)
        df = df[df["sector"].notna() & (df["sector"] != "")]
        agg = df.groupby(["sector", "option_type"]).agg(
            total_premium=("premium", "sum"),
            total_volume=("size", "sum"),
            trade_count=("size", "count"),
            avg_iv=("implied_volatility", "mean"),
        ).reset_index()
        # Pivot for readability
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
        df = _load_options(date)
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
        total_col = "total_premium_call" if "total_premium_call" in pivot.columns else pivot.columns[1]
        pivot["total"] = pivot.get("total_premium_call", 0) + pivot.get("total_premium_put", 0)
        pivot = pivot.sort_values("total", ascending=False).head(top_n)
        return df_to_result(pivot)

    elif name == "greek_screener":
        df = _load_options(date)
        if arguments.get("symbol"):
            df = df[df["underlying_symbol"] == arguments["symbol"].upper()]
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

    elif name == "iv_term_structure":
        symbol = (arguments.get("symbol") or "").upper().strip() or None
        iv_cols = ["underlying_symbol", "expiry", "implied_volatility"]
        df = _load_options(date, usecols=iv_cols)
        if symbol:
            df = df[df["underlying_symbol"] == symbol]
        df = df.copy()
        df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
        df = df[df["implied_volatility"] > 0]
        if df.empty:
            return text_result({"error": f"No options data{f' for {symbol}' if symbol else ''}"})
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df = df.dropna(subset=["expiry"])

        agg = (
            df.groupby("expiry")
            .agg(avg_iv=("implied_volatility", "mean"), contract_count=("implied_volatility", "count"))
            .reset_index()
            .sort_values("expiry")
        )

        today = date_cls.today()
        agg["dte_approx"] = (agg["expiry"].dt.date - today).apply(lambda d: d.days)

        iv_values = agg["avg_iv"].to_numpy()
        n = len(agg)
        kink_expiry = None

        if n < 2:
            structure = "INSUFFICIENT_DATA"
        else:
            first_iv, last_iv = float(iv_values[0]), float(iv_values[-1])
            if first_iv > last_iv * 1.05:
                structure = "BACKWARDATION"
            elif last_iv > first_iv * 1.05:
                structure = "CONTANGO"
            elif n >= 4:
                x = np.arange(n, dtype=float)
                coeffs = np.polyfit(x, iv_values, 1)
                fitted = np.polyval(coeffs, x)
                safe_fitted = np.where(np.abs(fitted) > 0.001, np.abs(fitted), 0.001)
                residuals = np.abs(iv_values - fitted) / safe_fitted
                max_idx = int(np.argmax(residuals))
                if residuals[max_idx] > 0.15:
                    structure = "KINKED"
                    kink_expiry = str(agg.iloc[max_idx]["expiry"].date())
                else:
                    structure = "FLAT"
            else:
                structure = "FLAT"

        term_structure = [
            {
                "expiry": str(row["expiry"].date()),
                "avg_iv": round(float(row["avg_iv"]), 4),
                "avg_iv_pct": f"{row['avg_iv'] * 100:.1f}%",
                "dte_approx": int(row["dte_approx"]),
                "contract_count": int(row["contract_count"]),
            }
            for _, row in agg.iterrows()
        ]

        return text_result({
            "symbol": symbol or "all",
            "structure": structure,
            "kink_expiry": kink_expiry,
            "expiry_count": n,
            "term_structure": term_structure,
        })

    elif name == "gamma_exposure_profile":
        symbol = arguments["symbol"].upper().strip()
        include_zero_gamma = arguments.get("include_zero_gamma", True)
        gex_cols = ["underlying_symbol", "strike", "option_type", "gamma", "open_interest", "underlying_price"]
        df = _load_options(date, usecols=gex_cols)
        df = df[df["underlying_symbol"] == symbol].copy()
        if df.empty:
            return text_result({"error": f"No options data for {symbol}"})

        df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
        df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df = df.dropna(subset=["gamma", "open_interest", "underlying_price", "strike"])
        df = df[(df["gamma"] > 0) & (df["open_interest"] > 0)]
        if df.empty:
            return text_result({"error": f"No valid gamma/OI data for {symbol}"})

        underlying_price = float(df["underlying_price"].median())

        # GEX per row: positive for calls, negative for puts (dealer perspective)
        df["gex"] = df["gamma"] * df["open_interest"] * 100 * df["underlying_price"]
        df.loc[df["option_type"] == "put", "gex"] = -df.loc[df["option_type"] == "put", "gex"]

        strike_gex = (
            df.groupby("strike")["gex"]
            .sum()
            .reset_index()
            .sort_values("strike")
            .reset_index(drop=True)
        )
        total_gex = float(strike_gex["gex"].sum())

        zero_gamma_level = None
        if include_zero_gamma:
            strike_gex["cumulative_gex"] = strike_gex["gex"].cumsum()
            for i in range(1, len(strike_gex)):
                g1 = float(strike_gex.iloc[i - 1]["cumulative_gex"])
                g2 = float(strike_gex.iloc[i]["cumulative_gex"])
                if g1 * g2 < 0:
                    s1 = float(strike_gex.iloc[i - 1]["strike"])
                    s2 = float(strike_gex.iloc[i]["strike"])
                    zero_gamma_level = round(s1 + (s2 - s1) * (-g1) / (g2 - g1), 2)
                    break

        if zero_gamma_level is None:
            regime = "FULLY_POSITIVE" if total_gex > 0 else "FULLY_NEGATIVE"
        else:
            regime = "POSITIVE" if underlying_price > zero_gamma_level else "NEGATIVE"

        regime_desc = {
            "POSITIVE": "Dealers net long gamma — expect mean-reversion and reduced volatility",
            "NEGATIVE": "Dealers net short gamma — expect trend acceleration and increased volatility",
            "FULLY_POSITIVE": "All strikes have positive net GEX — strong gamma pinning effect",
            "FULLY_NEGATIVE": "All strikes have negative net GEX — strong gamma amplification",
        }[regime]

        per_strike = [
            {"strike": float(r["strike"]), "net_gex": round(float(r["gex"]), 2)}
            for _, r in strike_gex.iterrows()
        ]

        return text_result({
            "symbol": symbol,
            "underlying_price": round(underlying_price, 2),
            "total_gex": round(total_gex, 0),
            "zero_gamma_level": zero_gamma_level,
            "regime": regime,
            "regime_description": regime_desc,
            "per_strike": per_strike[:50],
            "note": "GEX most meaningful for index products (SPY, QQQ) and large-cap single stocks with deep OI.",
        })

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
