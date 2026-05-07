"""MCP Server: per-ticker structural options analytics.

Phase 2 split: holds the structural per-symbol tools (Greeks, IV term-structure,
GEX, DEX, vanna/charm) separated from the trade-level flow tools in `uw-options-flow`.

Tools:
  - iv_term_structure        BACKWARDATION/CONTANGO/KINKED classifier
  - term_skew                25Δ put/call IV skew at target DTE
  - front_end_iv_ratio       Single-number near/far IV ratio
  - today_gamma_flip         0DTE-only zero-gamma + walls
  - gamma_exposure_profile   Per-strike GEX + zero-gamma level (with dte_max)
  - dealer_delta_exposure    Net dealer delta hedge requirement
  - vanna_charm_exposure     Net vanna and charm — vanna-squeeze setup detection
"""

from __future__ import annotations

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

from shared.analysis_gex import (
    classify_gex_regime,
    compute_gex_per_strike,
    compute_zero_gamma_level,
)
from shared.analysis_greeks import compute_charm_series, compute_vanna_series
from shared.data_loader import load_data
from shared.option_symbol import extract_option_expiries
from shared.server_utils import text_result

server = Server("uw-options-structure")


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
            name="term_skew",
            description=(
                "Compute back-month put/call skew at a target DTE. Returns the IV at "
                "~25-delta puts vs ~25-delta calls. Put IV >> call IV = market pricing tail risk. "
                "Skew at multi-month low can flag complacency; skew at high flags hedging demand."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (required)"},
                    "dte_target": {"type": "integer", "description": "Target DTE for skew snapshot (default: 365)", "default": 365},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="today_gamma_flip",
            description=(
                "Filter gamma_exposure_profile to today's expiry only. Returns the 0DTE "
                "zero-gamma level, ATM gamma flip strike, and key support/resistance walls — "
                "the precise dealer-hedging map for intraday/0DTE trading."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (required)"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="front_end_iv_ratio",
            description=(
                "Compute the ratio of near-term IV to back-end IV. Ratio > 1.05 = BACKWARDATION "
                "(panic / event imminent). Ratio < 0.95 = CONTANGO. Derived from the same data "
                "as iv_term_structure but returns a single tradeable number."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (required)"},
                    "near_dte": {"type": "integer", "description": "Near-term DTE target (default: 1)", "default": 1},
                    "far_dte": {"type": "integer", "description": "Back-end DTE target (default: 30)", "default": 30},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="gamma_exposure_profile",
            description=(
                "Calculate net dealer Gamma Exposure (GEX) per strike for a symbol. "
                "Positive GEX = dealers long gamma, price-stabilizing (pinning). "
                "Negative GEX = dealers short gamma, trend-amplifying. Returns the Zero "
                "Gamma Level where dealer hedging pressure flips. By default filters to "
                "DTE <= 45 to avoid LEAP-bias on the ZGL (audit §2.2)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (required — GEX is per-underlying)"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "dte_max": {
                        "type": "integer",
                        "description": "Filter to contracts with DTE <= this many days (default: 45). Set to a large value (e.g., 365) to include LEAPs.",
                        "default": 45,
                    },
                    "include_zero_gamma": {
                        "type": "boolean",
                        "description": "Compute zero gamma flip level (default: true)",
                        "default": True,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="vanna_charm_exposure",
            description=(
                "Net vanna and charm per ticker. Vanna = sensitivity of delta to vol; "
                "Charm = sensitivity of delta to time. Heavy negative net vanna + falling "
                "VIX is the classic vanna squeeze setup (Karsan, SpotGamma 2021). "
                "Charm flows accelerate around OPEX week. Closed-form approximations "
                "from stored Greeks — coarse for deep ITM/OTM and 0DTE; filter dte >= 1 "
                "by default. Sign convention is public-perspective (calls negative vanna, "
                "puts positive vanna); dealer hedge is the inverse."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (required)"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "dte_max": {
                        "type": "integer",
                        "description": "Filter to contracts with DTE <= this many days (default: 45). Set higher to include LEAPs.",
                        "default": 45,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="dealer_delta_exposure",
            description=(
                "Net dealer delta hedge requirement per ticker. Sum of (delta × OI × 100 × spot) "
                "across the OI book. Positive net DEX = call-heavy public positioning → dealers "
                "are net short calls → dealer hedge is to BUY underlying. Negative net DEX = "
                "put-heavy positioning → dealer hedge is to SELL underlying. DEX flips often "
                "precede directional moves (Karsan, SqueezeMetrics). Defaults to DTE <= 45 to "
                "focus on near-term hedging like GEX."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (required)"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "dte_max": {
                        "type": "integer",
                        "description": "Filter to contracts with DTE <= this many days (default: 45). Set higher (e.g., 365) to include LEAPs.",
                        "default": 45,
                    },
                },
                "required": ["symbol"],
            },
        ),
    ]


# ============================================================================
# Helpers
# ============================================================================


def _gex_input_columns() -> list[str]:
    return [
        "underlying_symbol", "strike", "option_type", "gamma",
        "open_interest", "underlying_price", "expiry",
    ]


def _dex_input_columns() -> list[str]:
    return [
        "underlying_symbol", "strike", "option_type", "delta",
        "open_interest", "underlying_price", "expiry",
    ]


# ============================================================================
# Dispatch
# ============================================================================


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    date = arguments.get("date")

    if name == "iv_term_structure":
        symbol = (arguments.get("symbol") or "").upper().strip() or None
        iv_cols = ["underlying_symbol", "expiry", "implied_volatility"]
        try:
            df = _load_options(date, usecols=iv_cols)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
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

    elif name == "term_skew":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper().strip()
        dte_target = int(arguments.get("dte_target", 365))
        cols = ["underlying_symbol", "expiry", "option_type", "delta", "implied_volatility"]
        try:
            df = _load_options(date, usecols=cols)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        df = df[df["underlying_symbol"] == symbol].copy()
        if df.empty:
            return text_result({"symbol": symbol, "note": "no options data"})
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
        df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
        df = df.dropna(subset=["expiry", "delta", "implied_volatility"])
        df = df[df["implied_volatility"] > 0]
        if df.empty:
            return text_result({"symbol": symbol, "note": "no valid skew data"})
        today = date_cls.today()
        df["dte"] = (df["expiry"].dt.date - today).apply(lambda d: d.days)
        df = df[df["dte"] >= 0]
        if df.empty:
            return text_result({"symbol": symbol, "note": "no future expiries"})
        df["dte_distance"] = (df["dte"] - dte_target).abs()
        nearest_dte = int(df.sort_values("dte_distance").iloc[0]["dte"])
        snap = df[df["dte"] == nearest_dte].copy()
        snap["abs_delta"] = snap["delta"].abs()
        snap["delta_distance"] = (snap["abs_delta"] - 0.25).abs()
        calls = snap[snap["option_type"] == "call"].sort_values("delta_distance").head(5)
        puts = snap[snap["option_type"] == "put"].sort_values("delta_distance").head(5)
        if calls.empty or puts.empty:
            return text_result({"symbol": symbol, "note": "insufficient call/put coverage at target DTE"})
        call_iv = float(calls["implied_volatility"].mean())
        put_iv = float(puts["implied_volatility"].mean())
        skew = put_iv - call_iv
        skew_ratio = put_iv / call_iv if call_iv > 0 else 0.0
        if skew_ratio > 1.10:
            interpretation = "TAIL_HEDGING"
        elif skew_ratio < 1.02:
            interpretation = "COMPLACENT"
        else:
            interpretation = "NORMAL"
        return text_result({
            "symbol": symbol,
            "dte_target": dte_target,
            "dte_actual": nearest_dte,
            "call_25d_iv": round(call_iv, 4),
            "put_25d_iv": round(put_iv, 4),
            "skew": round(skew, 4),
            "skew_ratio": round(skew_ratio, 3),
            "interpretation": interpretation,
        })

    elif name == "today_gamma_flip":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper().strip()
        try:
            df = _load_options(date, usecols=_gex_input_columns())
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        df = df[df["underlying_symbol"] == symbol].copy()
        if df.empty:
            return text_result({"symbol": symbol, "note": "no options data"})
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df = df.dropna(subset=["expiry"])
        if df.empty:
            return text_result({"symbol": symbol, "note": "no valid expiry data"})
        today_expiry = df["expiry"].dt.date.min()
        df = df[df["expiry"].dt.date == today_expiry]
        if df.empty:
            return text_result({"symbol": symbol, "note": "no 0DTE data"})

        # Compute GEX via shared module (no DTE filter — already pre-filtered to today's expiry)
        df["expiry"] = df["expiry"].dt.strftime("%Y-%m-%d")
        strike_gex = compute_gex_per_strike(df, dte_max=None)
        if strike_gex.empty:
            return text_result({"symbol": symbol, "note": "no valid 0DTE gamma/OI"})

        spot = float(pd.to_numeric(df["underlying_price"], errors="coerce").median())
        total_gex = float(strike_gex["gex"].sum())
        zero_gamma = compute_zero_gamma_level(strike_gex)

        # ATM flip strike: nearest sign-flip in per-strike GEX (not cumulative)
        atm_flip = None
        sg = strike_gex.reset_index(drop=True)
        for i in range(1, len(sg)):
            if float(sg.iloc[i - 1]["gex"]) * float(sg.iloc[i]["gex"]) < 0:
                cand_a = float(sg.iloc[i]["strike"])
                cand_b = float(sg.iloc[i - 1]["strike"])
                atm_flip = cand_a if abs(cand_a - spot) < abs(cand_b - spot) else cand_b
                break

        # today_gamma_flip preserves the original binary contract: POSITIVE or
        # NEGATIVE based on total_gex sign. The richer FULLY_*/FLAT vocabulary
        # belongs to gamma_exposure_profile only.
        regime = "POSITIVE" if total_gex > 0 else "NEGATIVE"

        key_walls = []
        for _, r in strike_gex.iterrows():
            gex_val = float(r["gex"])
            strike = float(r["strike"])
            if gex_val > 0 and strike < spot:
                role = "support_wall"
            elif gex_val < 0 and strike > spot:
                role = "resistance_wall"
            elif gex_val > 0:
                role = "support_wall"
            else:
                role = "resistance_wall"
            key_walls.append({"strike": strike, "gex": round(gex_val, 0), "role": role})
        key_walls = sorted(key_walls, key=lambda w: abs(w["gex"]), reverse=True)[:5]

        return text_result({
            "symbol": symbol,
            "spot": round(spot, 2),
            "today_expiry": str(today_expiry),
            "today_zero_gamma": zero_gamma,
            "atm_flip_strike": atm_flip,
            "today_total_gex": round(total_gex, 0),
            "regime": regime,
            "key_walls": key_walls,
        })

    elif name == "front_end_iv_ratio":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper().strip()
        near_dte = int(arguments.get("near_dte", 1))
        far_dte = int(arguments.get("far_dte", 30))
        if near_dte >= far_dte:
            raise ValueError(
                f"near_dte ({near_dte}) must be strictly less than far_dte ({far_dte})"
            )
        cols = ["underlying_symbol", "expiry", "implied_volatility"]
        try:
            df = _load_options(date, usecols=cols)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        df = df[df["underlying_symbol"] == symbol].copy()
        if df.empty:
            return text_result({"symbol": symbol, "note": "no options data"})
        df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df = df.dropna(subset=["implied_volatility", "expiry"])
        df = df[df["implied_volatility"] > 0]
        today = date_cls.today()
        df["dte"] = (df["expiry"].dt.date - today).apply(lambda d: d.days)
        df = df[df["dte"] >= 0]
        if df.empty:
            return text_result({"symbol": symbol, "note": "no future expiries"})
        df["near_distance"] = (df["dte"] - near_dte).abs()
        df["far_distance"] = (df["dte"] - far_dte).abs()
        nearest_near_dte = int(df.sort_values("near_distance").iloc[0]["dte"])
        nearest_far_dte = int(df.sort_values("far_distance").iloc[0]["dte"])
        near_iv = float(df[df["dte"] == nearest_near_dte]["implied_volatility"].mean())
        far_iv = float(df[df["dte"] == nearest_far_dte]["implied_volatility"].mean())
        ratio = near_iv / far_iv if far_iv > 0 else 0.0
        if ratio > 1.05:
            regime = "BACKWARDATION"
        elif ratio < 0.95:
            regime = "CONTANGO"
        else:
            regime = "FLAT"
        return text_result({
            "symbol": symbol,
            "near_dte_actual": nearest_near_dte,
            "far_dte_actual": nearest_far_dte,
            "near_iv": round(near_iv, 4),
            "far_iv": round(far_iv, 4),
            "ratio": round(ratio, 3),
            "regime": regime,
        })

    elif name == "gamma_exposure_profile":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper().strip()
        include_zero_gamma = arguments.get("include_zero_gamma", True)
        dte_max = arguments.get("dte_max", 45)
        if dte_max is not None and (dte_max < 1 or dte_max > 1500):
            raise ValueError(f"dte_max must be 1-1500 or null, got {dte_max}")

        try:
            df = _load_options(date, usecols=_gex_input_columns())
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        df = df[df["underlying_symbol"] == symbol].copy()
        if df.empty:
            return text_result({"error": f"No options data for {symbol}"})

        # Normalise expiry to ISO string for shared.analysis_gex
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")

        strike_gex = compute_gex_per_strike(df, dte_max=dte_max)
        if strike_gex.empty:
            return text_result({"error": f"No valid gamma/OI data for {symbol} within dte_max={dte_max}"})

        underlying_price = float(pd.to_numeric(df["underlying_price"], errors="coerce").median())
        total_gex = float(strike_gex["gex"].sum())
        zero_gamma_level = compute_zero_gamma_level(strike_gex) if include_zero_gamma else None
        regime = classify_gex_regime(total_gex, underlying_price, zero_gamma_level)

        regime_desc = {
            "POSITIVE": "Dealers net long gamma — expect mean-reversion and reduced volatility",
            "NEGATIVE": "Dealers net short gamma — expect trend acceleration and increased volatility",
            "FULLY_POSITIVE": "All strikes have positive net GEX — strong gamma pinning effect",
            "FULLY_NEGATIVE": "All strikes have negative net GEX — strong gamma amplification",
            "FLAT": "Total GEX near zero — no clear hedging pressure",
        }.get(regime, "")

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
            "dte_max": dte_max,
            "per_strike": per_strike[:50],
            "note": "GEX most meaningful for index products (SPY, QQQ) and large-cap single stocks with deep OI.",
        })

    elif name == "vanna_charm_exposure":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper().strip()
        dte_max_raw = arguments.get("dte_max", 45)
        dte_max = int(dte_max_raw) if dte_max_raw is not None else None
        if dte_max is not None and (dte_max < 1 or dte_max > 1500):
            raise ValueError(f"dte_max must be 1-1500 or null, got {dte_max}")

        cols = [
            "underlying_symbol", "option_type", "delta", "vega", "theta",
            "implied_volatility", "underlying_price", "price",
            "open_interest", "expiry",
        ]
        try:
            df = _load_options(date, usecols=cols)
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        df = df[df["underlying_symbol"] == symbol].copy()
        if df.empty:
            return text_result({"error": f"No options data for {symbol}"})

        # Coerce + drop rows missing the Greeks we need
        for col in ("delta", "vega", "theta", "implied_volatility",
                    "underlying_price", "price", "open_interest"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df = df.dropna(subset=["delta", "vega", "theta", "implied_volatility",
                                "underlying_price", "price", "open_interest", "expiry"])
        df = df[df["open_interest"] > 0]

        # Sign convention: BSM-natural delta. Defensive guard against put delta
        # being stored as positive by some data providers (matches dex tool).
        put_mask = df["option_type"] == "put"
        df.loc[put_mask & (df["delta"] > 0), "delta"] = (
            -df.loc[put_mask & (df["delta"] > 0), "delta"]
        )

        # DTE filter — exclude expired (dte<0) AND optionally far-dated
        today = pd.Timestamp(date_cls.today())
        df["dte"] = (df["expiry"] - today).dt.days
        df = df[df["dte"] >= 1]  # 0DTE excluded — approximations break down at expiry
        if dte_max is not None:
            df = df[df["dte"] <= dte_max]
        if df.empty:
            return text_result({
                "error": f"No contracts within dte_max={dte_max} for {symbol} (after dte>=1 filter)"
            })

        # Per-row Greeks weighted by OI × 100 (consistent with GEX/DEX scaling)
        vanna_per_row = compute_vanna_series(df) * df["open_interest"] * 100
        charm_per_row = compute_charm_series(df) * df["open_interest"] * 100

        net_vanna = float(vanna_per_row.sum())
        net_charm = float(charm_per_row.sum())
        call_vanna = float(vanna_per_row[df["option_type"] == "call"].sum())
        put_vanna = float(vanna_per_row[df["option_type"] == "put"].sum())
        spot = float(df["underlying_price"].median())

        if net_vanna < 0:
            vanna_interp = (
                "Public net vanna negative (call-heavy book). Falling IV → "
                "call delta drops → dealers (short calls) cut long-underlying hedge → "
                "SELLING pressure. Rising IV reverses the flow."
            )
        elif net_vanna > 0:
            vanna_interp = (
                "Public net vanna positive (put-heavy book). Falling IV → "
                "|put delta| drops → dealers (short puts) cover by BUYING underlying. "
                "Classic vanna-squeeze setup if VIX collapses."
            )
        else:
            vanna_interp = "Net vanna balanced."

        return text_result({
            "symbol": symbol,
            "spot": round(spot, 2),
            "net_vanna": round(net_vanna, 0),
            "call_vanna": round(call_vanna, 0),
            "put_vanna": round(put_vanna, 0),
            "net_charm": round(net_charm, 0),
            "dte_max": dte_max,
            "vanna_interpretation": vanna_interp,
            "note": (
                "Closed-form approximations: vanna ≈ -vega × delta / (σ × S); "
                "charm ≈ -theta × delta / option_price. Coarse for deep ITM/OTM "
                "and near-expiry — filtered to dte >= 1 by default."
            ),
        })

    elif name == "dealer_delta_exposure":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper().strip()
        dte_max = arguments.get("dte_max", 45)
        if dte_max is not None and (dte_max < 1 or dte_max > 1500):
            raise ValueError(f"dte_max must be 1-1500 or null, got {dte_max}")

        try:
            df = _load_options(date, usecols=_dex_input_columns())
        except FileNotFoundError:
            return text_result({"error": "No data available for this date"})
        df = df[df["underlying_symbol"] == symbol].copy()
        if df.empty:
            return text_result({"error": f"No options data for {symbol}"})

        df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
        df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df = df.dropna(subset=["delta", "open_interest", "underlying_price", "expiry"])
        df = df[df["open_interest"] > 0]
        if df.empty:
            return text_result({"error": f"No valid delta/OI data for {symbol}"})

        # Sign convention: BSM-natural delta — puts must be stored as negative.
        # Defensive guard against data-provider sign drift (audit hardening).
        put_mask = df["option_type"] == "put"
        df.loc[put_mask & (df["delta"] > 0), "delta"] = (
            -df.loc[put_mask & (df["delta"] > 0), "delta"]
        )

        # DTE filter (same convention as GEX)
        if dte_max is not None:
            today = pd.Timestamp(date_cls.today())
            df["dte"] = (df["expiry"] - today).dt.days
            df = df[(df["dte"] >= 0) & (df["dte"] <= dte_max)]
            if df.empty:
                return text_result({
                    "error": f"No contracts within dte_max={dte_max} for {symbol}"
                })

        # Per-row DEX = delta × OI × 100 × spot (natural sign of delta:
        # calls contribute positively, puts contribute negatively)
        df["dex_row"] = df["delta"] * df["open_interest"] * 100 * df["underlying_price"]
        call_dex = float(df.loc[df["option_type"] == "call", "dex_row"].sum())
        put_dex = float(df.loc[df["option_type"] == "put", "dex_row"].sum())
        net_dex = call_dex + put_dex
        spot = float(df["underlying_price"].median())

        if net_dex > 0:
            interpretation = (
                "Public is net call-long → dealers net short calls → "
                "dealer hedge is to BUY underlying."
            )
        elif net_dex < 0:
            interpretation = (
                "Public is net put-long → dealers net short puts → "
                "dealer hedge is to SELL underlying."
            )
        else:
            interpretation = "Net delta exposure balanced."

        return text_result({
            "symbol": symbol,
            "spot": round(spot, 2),
            "call_dex": round(call_dex, 0),
            "put_dex": round(put_dex, 0),
            "net_dex": round(net_dex, 0),
            "dte_max": dte_max,
            "interpretation": interpretation,
            "note": (
                "DEX = sum(delta × OI × 100 × spot). Sign reflects public positioning "
                "(positive = call-heavy). Dealer hedge is the inverse."
            ),
        })

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
