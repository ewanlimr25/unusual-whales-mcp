"""MCP Server: Unusual Whales Open Interest Changes analysis."""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import load_data
from shared.option_symbol import classify_option_types, extract_option_expiries
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
                    "min_dte": {"type": "integer", "description": "Filter to contracts with DTE >= min_dte (e.g. 180 for LEAPs only). Optional."},
                    "max_dte": {"type": "integer", "description": "Filter to contracts with DTE <= max_dte (e.g. 30 for near-term only). Optional."},
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
        Tool(
            name="pin_risk_screener",
            description=(
                "Rank tickers within N days of OPEX by likelihood of pinning at "
                "their highest-OI strike. Score = sum(curr_oi × proximity_weight) "
                "across strikes within max_distance_pct of spot, where "
                "proximity_weight = exp(-|distance|/spot/0.02). Higher score = "
                "stronger pin candidate. Per Ni-Pearson-Poteshman 2005 / "
                "Avellaneda-Lipkin 2003. Single-day OI snapshot."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of tickers (default: 20)", "default": 20},
                    "dte_max": {
                        "type": "integer",
                        "description": "Only include tickers with min DTE <= this (default: 7 = OPEX week)",
                        "default": 7,
                    },
                    "max_distance_pct": {
                        "type": "number",
                        "description": "Strike must be within this % of spot to count toward score (default: 5)",
                        "default": 5.0,
                    },
                    "min_total_oi": {
                        "type": "integer",
                        "description": "Minimum aggregated OI per ticker (default: 1000)",
                        "default": 1000,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="opex_concentration",
            description=(
                "For each ticker, find the expiry holding the largest fraction of total OI. "
                "High concentration (>50%) signals an OPEX cliff — concentrated dealer hedging "
                "around that expiry. When spot is within 5% of the highest-OI strike at the "
                "concentrated expiry, returns pin_distance_pct for the Stoll-Whaley / "
                "Ni-Pearson-Poteshman pinning effect. Single-day snapshot only."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "top_n": {"type": "integer", "description": "Number of tickers to return (default: 20)", "default": 20},
                    "min_concentration_pct": {
                        "type": "number",
                        "description": "Minimum concentration percentage to include (default: 50)",
                        "default": 50.0,
                    },
                    "min_total_oi": {
                        "type": "integer",
                        "description": "Minimum total OI per ticker to filter noise (default: 1000)",
                        "default": 1000,
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
        if arguments.get("min_dte") is not None or arguments.get("max_dte") is not None:
            df = df.copy()
            df["dte"] = pd.to_numeric(df["dte"], errors="coerce")
            df = df.dropna(subset=["dte"])
            if arguments.get("min_dte") is not None:
                df = df[df["dte"] >= arguments["min_dte"]]
            if arguments.get("max_dte") is not None:
                df = df[df["dte"] <= arguments["max_dte"]]
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
        # Infer if this is a call or put from the option symbol (OPRA position-aware)
        df = df.copy()
        df["option_type_inferred"] = classify_option_types(df["option_symbol"])
        df = df.dropna(subset=["option_type_inferred"])
        df["is_call"] = df["option_type_inferred"] == "call"
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

        # Infer option_type from option_symbol (OPRA position-aware C/P parser)
        df["option_type_inferred"] = classify_option_types(df["option_symbol"])
        df = df.dropna(subset=["option_type_inferred"])
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

    elif name == "pin_risk_screener":
        top_n = int(arguments.get("top_n", 20))
        dte_max = int(arguments.get("dte_max", 7))
        max_distance_pct = float(arguments.get("max_distance_pct", 5.0))
        min_total_oi = int(arguments.get("min_total_oi", 1000))

        if top_n < 1 or top_n > 500:
            raise ValueError(f"top_n must be 1-500, got {top_n}")
        # dte_max=0 is a valid OPEX-Friday query (peak gamma at expiry).
        # Cap at 365 — beyond a year is not "near OPEX" by any defensible reading.
        if dte_max < 0 or dte_max > 365:
            raise ValueError(f"dte_max must be 0-365, got {dte_max}")
        if max_distance_pct <= 0 or max_distance_pct > 50:
            raise ValueError(
                f"max_distance_pct must be in (0, 50], got {max_distance_pct}"
            )
        if min_total_oi < 0:
            raise ValueError(f"min_total_oi must be non-negative, got {min_total_oi}")

        try:
            df = load_data(
                "oi",
                date,
                usecols=["underlying_symbol", "strike", "curr_oi", "stock_price", "dte"],
            )
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})

        df = df.copy()
        df["curr_oi"] = pd.to_numeric(df["curr_oi"], errors="coerce").fillna(0)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["stock_price"] = pd.to_numeric(df["stock_price"], errors="coerce")
        df["dte"] = pd.to_numeric(df["dte"], errors="coerce")
        df = df.dropna(subset=["strike", "stock_price", "dte"])
        df = df[df["curr_oi"] > 0]
        df = df[df["dte"] >= 0]

        # Per-ticker minimum DTE — must be within OPEX window
        per_ticker_min_dte = df.groupby("underlying_symbol")["dte"].min()
        eligible = per_ticker_min_dte[per_ticker_min_dte <= dte_max].index
        df = df[df["underlying_symbol"].isin(eligible)]
        if df.empty:
            return text_result({
                "caveat": "Single-day OI snapshot. Pin risk evaluated only for tickers with min DTE within window.",
                "results": [],
            })

        # Per-strike distance from spot (per row)
        df["distance_pct"] = (df["strike"] - df["stock_price"]).abs() / df["stock_price"] * 100
        df = df[df["distance_pct"] <= max_distance_pct]
        if df.empty:
            return text_result({
                "caveat": "Single-day OI snapshot.",
                "results": [],
            })

        # Proximity weight: exp(-distance_decimal / 0.02). At 0% distance weight=1;
        # at 2% distance weight≈0.37; at 5% weight≈0.08.
        df["proximity_weight"] = np.exp(-df["distance_pct"] / 100.0 / 0.02)
        df["score_contribution"] = df["curr_oi"] * df["proximity_weight"]

        # Aggregate per ticker
        agg = df.groupby("underlying_symbol", as_index=False).agg(
            pin_score=("score_contribution", "sum"),
            total_oi_in_window=("curr_oi", "sum"),
            spot=("stock_price", "median"),
            dte_to_opex=("dte", "min"),
        )
        agg = agg[agg["total_oi_in_window"] >= min_total_oi]

        # For each ticker find the strike with the highest curr_oi within window
        idx_max = df.groupby("underlying_symbol")["curr_oi"].idxmax()
        top_strikes = df.loc[idx_max, ["underlying_symbol", "strike", "curr_oi", "distance_pct"]]
        top_strikes = top_strikes.rename(columns={
            "strike": "nearest_high_oi_strike",
            "curr_oi": "top_strike_oi",
            "distance_pct": "pin_distance_pct",
        })
        merged = agg.merge(top_strikes, on="underlying_symbol", how="left")
        merged = merged.sort_values("pin_score", ascending=False).head(top_n)

        results = [
            {
                "ticker": row["underlying_symbol"],
                "spot": round(float(row["spot"]), 2),
                "nearest_high_oi_strike": round(float(row["nearest_high_oi_strike"]), 2),
                "pin_distance_pct": round(float(row["pin_distance_pct"]), 2),
                "top_strike_oi": int(row["top_strike_oi"]),
                "total_oi_in_window": int(row["total_oi_in_window"]),
                "dte_to_opex": int(row["dte_to_opex"]),
                "pin_score": round(float(row["pin_score"]), 2),
            }
            for _, row in merged.iterrows()
        ]

        return text_result({
            "caveat": (
                "Single-day OI snapshot. Pin risk reflects gamma-weighted OI mass near spot "
                "for tickers in OPEX window (min DTE <= dte_max). Higher pin_score = stronger "
                "pinning candidate. Gamma weighting is implicit (proximity_weight is highest "
                "at-the-money, where gamma is highest at near-expiry)."
            ),
            "dte_max": dte_max,
            "max_distance_pct": max_distance_pct,
            "results": results,
        })

    elif name == "opex_concentration":
        top_n = int(arguments.get("top_n", 20))
        min_concentration_pct = float(arguments.get("min_concentration_pct", 50.0))
        min_total_oi = int(arguments.get("min_total_oi", 1000))

        if top_n < 1 or top_n > 500:
            raise ValueError(f"top_n must be 1-500, got {top_n}")
        if not (0.0 < min_concentration_pct <= 100.0):
            raise ValueError(
                f"min_concentration_pct must be in (0, 100], got {min_concentration_pct}"
            )
        if min_total_oi < 0:
            raise ValueError(f"min_total_oi must be non-negative, got {min_total_oi}")

        try:
            df = load_data(
                "oi",
                date,
                usecols=["option_symbol", "underlying_symbol", "curr_oi", "strike", "stock_price"],
            )
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})

        df = df.copy()
        df["curr_oi"] = pd.to_numeric(df["curr_oi"], errors="coerce").fillna(0)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["stock_price"] = pd.to_numeric(df["stock_price"], errors="coerce")
        df["expiry"] = extract_option_expiries(df["option_symbol"])
        df = df.dropna(subset=["expiry", "strike"])
        df = df[df["curr_oi"] > 0]

        if df.empty:
            return text_result({
                "caveat": "Single-day OI snapshot. Concentrated expiry = the expiry holding the largest OI fraction per ticker.",
                "results": [],
            })

        # Per-ticker total OI
        per_ticker_total = df.groupby("underlying_symbol")["curr_oi"].sum()
        eligible_tickers = per_ticker_total[per_ticker_total >= min_total_oi].index
        df = df[df["underlying_symbol"].isin(eligible_tickers)]

        # Per-ticker × expiry OI; identify the expiry with max OI per ticker
        per_expiry = (
            df.groupby(["underlying_symbol", "expiry"])["curr_oi"]
            .sum()
            .reset_index()
        )
        idx_max = per_expiry.groupby("underlying_symbol")["curr_oi"].idxmax()
        top_expiries = per_expiry.loc[idx_max].rename(
            columns={"curr_oi": "concentrated_expiry_oi"}
        )

        results: list[dict] = []
        for _, row in top_expiries.iterrows():
            ticker = row["underlying_symbol"]
            expiry = row["expiry"]
            total_oi = float(per_ticker_total[ticker])
            concentrated_oi = float(row["concentrated_expiry_oi"])
            concentration_pct = (concentrated_oi / total_oi) * 100 if total_oi > 0 else 0.0
            if concentration_pct < min_concentration_pct:
                continue

            # Highest-OI strike at the concentrated expiry
            strike_slice = df[
                (df["underlying_symbol"] == ticker) & (df["expiry"] == expiry)
            ]
            top_strike_row = strike_slice.loc[strike_slice["curr_oi"].idxmax()]
            top_strike = float(top_strike_row["strike"])
            spot = float(top_strike_row["stock_price"]) if pd.notna(top_strike_row["stock_price"]) else None

            pin_distance_pct = None
            if spot and spot > 0:
                distance_pct = abs(top_strike - spot) / spot * 100
                if distance_pct <= 5.0:
                    pin_distance_pct = round(distance_pct, 2)

            results.append({
                "ticker": ticker,
                "concentrated_expiry": expiry,
                "concentration_pct": round(concentration_pct, 2),
                "concentrated_expiry_oi": int(concentrated_oi),
                "total_oi": int(total_oi),
                "top_strike": top_strike,
                "spot": round(spot, 2) if spot else None,
                "pin_distance_pct": pin_distance_pct,
            })

        results.sort(key=lambda r: r["concentration_pct"], reverse=True)
        results = results[:top_n]

        return text_result({
            "caveat": (
                "Single-day OI snapshot. Concentrated expiry = the expiry holding the "
                "largest OI fraction per ticker. pin_distance_pct only reported when spot "
                "is within 5% of the highest-OI strike at the concentrated expiry."
            ),
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
