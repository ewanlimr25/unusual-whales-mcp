"""MCP Server: Historical trend analysis across multiple days of Unusual Whales data."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from yfinance import Ticker

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.analysis_gex import (
    classify_gex_regime,
    compute_gex_per_strike,
    compute_zero_gamma_level,
)
from shared.analysis_historical import (
    iter_dates,
    percentile_rank,
    realised_volatility,
    zscore,
)
from shared.data_loader import list_available_dates, load_data
from shared.option_symbol import extract_option_expiries
from shared.server_utils import df_to_result, text_result

# Per-process cache for gex_time_series.
# Keyed by (symbol, file_date, dte_max) where dte_max=None means "no filter".
# Always int-or-None — cast at the dispatch layer before this is called.
_GEX_DAILY_CACHE: dict[tuple[str, str, int | None], dict] = {}


def _compute_gex_for_date(symbol: str, file_date: str, dte_max: int | None) -> dict | None:
    """Compute total GEX, ZGL, and spot for one symbol on one date.

    Memoised in `_GEX_DAILY_CACHE`. Returns None when no usable data exists
    (so callers can `continue` cleanly).

    `dte_max` semantics:
      - int (e.g. 45) → keep contracts with DTE in [0, dte_max]
      - None          → no DTE filter (LEAPs included)
    These are distinct cache keys; do not pass `None` thinking it means "default".
    """
    cache_key = (symbol, file_date, dte_max)
    if cache_key in _GEX_DAILY_CACHE:
        return _GEX_DAILY_CACHE[cache_key]

    cols = ["underlying_symbol", "strike", "option_type", "gamma",
            "open_interest", "underlying_price", "expiry"]
    try:
        df = load_data("options", file_date, usecols=cols)
    except FileNotFoundError:
        return None
    df = df[df["underlying_symbol"] == symbol].copy()
    if df.empty:
        return None
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
    strike_gex = compute_gex_per_strike(df, dte_max=dte_max)
    if strike_gex.empty:
        return None
    spot = float(pd.to_numeric(df["underlying_price"], errors="coerce").median())
    total_gex = float(strike_gex["gex"].sum())
    zgl = compute_zero_gamma_level(strike_gex)
    regime = classify_gex_regime(total_gex, spot, zgl)
    payload = {
        "spot": round(spot, 2),
        "total_gex": round(total_gex, 0),
        "zero_gamma_level": zgl,
        "regime": regime,
    }
    _GEX_DAILY_CACHE[cache_key] = payload
    return payload

server = Server("uw-historical")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="available_dates",
            description=(
                "List all available data dates for each data type. "
                "Use this first to see what historical data you have."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="trend_analyzer",
            description=(
                "Compare a ticker's options metrics across multiple days. "
                "Shows how volume, premium, IV rank, P/C ratio, and flow direction "
                "have been trending. Essential for spotting multi-day buildups."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "days": {"type": "integer", "description": "Number of recent days to analyze (default: all available)", "default": 0},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="oi_trend",
            description=(
                "Track open interest buildup or decline for a specific ticker "
                "across multiple days. Shows if positions are steadily building "
                "(conviction growing) or unwinding (thesis fading)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "days": {"type": "integer", "description": "Number of recent days (default: all available)", "default": 0},
                    "top_n": {"type": "integer", "description": "Top N contracts by OI change per day (default: 10)", "default": 10},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="cumulative_premium_flow",
            description=(
                "Sum bullish and bearish premium for a ticker across the most recent N sessions. "
                "Long-window cumulative flow is the LEAP-grade signature — slow accretion of net "
                "directional premium over weeks/months."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "days": {"type": "integer", "description": "Number of recent sessions to sum (default: 90)", "default": 90},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="pc_ratio_zscore",
            description=(
                "Z-score of a ticker's current put/call ratio vs its trailing window. "
                "|z| > 2 = sentiment extreme (BULLISH_EXTREME if ratio is unusually low, "
                "BEARISH_EXTREME if unusually high). Replaces raw P/C rank with a proper "
                "statistical measure."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "lookback_days": {"type": "integer", "description": "Window for mean/std (default: 20)", "default": 20},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="iv_percentile_zscore",
            description=(
                "Per-ticker IV30d percentile rank and z-score over a trailing N-day window. "
                "Outlier-robust replacement for the screener's IV-rank (which compresses on a "
                "single high reading). Goyal-Saretto 2009 — IV percentile is a cleaner sort. "
                "Returns: iv_percentile (0-100), iv_zscore, regime classification."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "lookback_days": {
                        "type": "integer",
                        "description": "Trailing window length (default: 252 = 1y trading days)",
                        "default": 252,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="gex_time_series",
            description=(
                "Multi-day Zero Gamma Level + total GEX trajectory for a symbol. "
                "Returns N daily ZGL values, total_gex per day, and a list of regime-flip dates "
                "where the ZGL crossed spot. Detects dealer-hedging regime transitions before "
                "realized vol expansion (Karsan / SpotGamma)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (required)"},
                    "days": {
                        "type": "integer",
                        "description": "Number of recent sessions (default: 30)",
                        "default": 30,
                    },
                    "dte_max": {
                        "type": "integer",
                        "description": "DTE cutoff per day passed to gamma_exposure_profile (default: 45)",
                        "default": 45,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="volatility_risk_premium",
            description=(
                "VRP = IV30d − realised σ(30d) for a ticker. Positive = options pricing more vol "
                "than realised → premium-selling environment. Negative = vol cheap vs realised → "
                "premium-buying environment. Bakshi-Kapadia 2003 / Carr-Wu 2009. Realised vol "
                "computed from Yahoo Finance close-to-close returns; gracefully handles yfinance "
                "outages."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD) for screener IV. Defaults to latest."},
                    "realised_window_days": {
                        "type": "integer",
                        "description": "Days of price history for realised σ (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="signal_backtest",
            description=(
                "Backtest a signal: when a ticker showed similar unusual activity in past data, "
                "what happened to the stock price in the following days? Uses Yahoo Finance "
                "for price verification. Requires multiple days of historical data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "signal_type": {
                        "type": "string",
                        "enum": ["bullish_flow", "bearish_flow", "high_iv_rank", "volume_spike", "dark_pool_accumulation"],
                        "description": "Type of signal to backtest",
                    },
                    "lookback_days": {"type": "integer", "description": "Days of price action to check after signal (default: 5)", "default": 5},
                    "top_n": {"type": "integer", "description": "Number of past signal instances to analyze (default: 20)", "default": 20},
                },
                "required": ["signal_type"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    if name == "available_dates":
        result = {}
        for dtype in ["screener", "options", "darkpool", "hotchains", "oi"]:
            result[dtype] = list_available_dates(dtype)
        return text_result(result)

    elif name == "trend_analyzer":
        symbol = arguments["symbol"].upper()
        dates = list_available_dates("screener")
        max_days = arguments.get("days", 0)
        if max_days > 0:
            dates = dates[:max_days]

        if not dates:
            return text_result("No screener data available")

        trend = []
        for date in dates:
            try:
                df = load_data("screener", date)
                row = df[df["ticker"] == symbol]
                if row.empty:
                    continue
                r = row.iloc[0]
                entry = {
                    "date": date,
                    "close": r.get("close"),
                    "call_volume": r.get("call_volume"),
                    "put_volume": r.get("put_volume"),
                    "call_premium": r.get("call_premium"),
                    "put_premium": r.get("put_premium"),
                    "put_call_ratio": r.get("put_call_ratio"),
                    "bullish_premium": r.get("bullish_premium"),
                    "bearish_premium": r.get("bearish_premium"),
                    "iv_rank": r.get("iv_rank"),
                    "iv30d": r.get("iv30d"),
                    "total_open_interest": r.get("total_open_interest"),
                }
                bull = float(r.get("bullish_premium", 0) or 0)
                bear = float(r.get("bearish_premium", 0) or 0)
                entry["net_flow"] = bull - bear
                entry["flow_direction"] = "bullish" if bull > bear else "bearish"
                trend.append(entry)
            except Exception:
                continue

        if not trend:
            return text_result(f"No historical data found for {symbol}")

        # Add trend summary
        if len(trend) >= 2:
            first = trend[-1]  # oldest
            last = trend[0]    # newest
            summary = {
                "symbol": symbol,
                "days_analyzed": len(trend),
                "date_range": f"{trend[-1]['date']} to {trend[0]['date']}",
                "price_change": f"{first.get('close')} -> {last.get('close')}",
                "iv_rank_change": f"{first.get('iv_rank')} -> {last.get('iv_rank')}",
                "flow_direction_latest": last.get("flow_direction"),
                "daily_data": trend,
            }
            # Count bullish vs bearish days
            bull_days = sum(1 for t in trend if t.get("flow_direction") == "bullish")
            summary["bullish_days"] = bull_days
            summary["bearish_days"] = len(trend) - bull_days
            return text_result(summary)

        return text_result({"symbol": symbol, "daily_data": trend})

    elif name == "oi_trend":
        symbol = arguments["symbol"].upper()
        dates = list_available_dates("oi")
        max_days = arguments.get("days", 0)
        if max_days > 0:
            dates = dates[:max_days]
        top_n = arguments.get("top_n", 10)

        if not dates:
            return text_result("No OI change data available")

        daily_oi = []
        for date in dates:
            try:
                df = load_data("oi", date)
                df = df[df["underlying_symbol"] == symbol]
                if df.empty:
                    continue
                df["oi_diff_plain"] = df["oi_diff_plain"].astype(float)
                total_increase = float(df[df["oi_diff_plain"] > 0]["oi_diff_plain"].sum())
                total_decrease = float(df[df["oi_diff_plain"] < 0]["oi_diff_plain"].sum())
                top_contracts = df.sort_values("oi_diff_plain", ascending=False).head(top_n)
                daily_oi.append({
                    "date": date,
                    "total_oi_increase": total_increase,
                    "total_oi_decrease": total_decrease,
                    "net_oi_change": total_increase + total_decrease,
                    "contracts_with_increases": int((df["oi_diff_plain"] > 0).sum()),
                    "contracts_with_decreases": int((df["oi_diff_plain"] < 0).sum()),
                    "top_contracts": json.loads(
                        top_contracts[["option_symbol", "strike", "oi_diff_plain", "volume", "dte"]]
                        .to_json(orient="records", default_handler=str)
                    ),
                })
            except Exception:
                continue

        if not daily_oi:
            return text_result(f"No OI data found for {symbol}")

        # Trend summary
        net_changes = [d["net_oi_change"] for d in daily_oi]
        result = {
            "symbol": symbol,
            "days_analyzed": len(daily_oi),
            "overall_trend": "BUILDING" if sum(net_changes) > 0 else "UNWINDING",
            "total_net_oi_change": sum(net_changes),
            "consecutive_build_days": 0,
            "daily_data": daily_oi,
        }
        # Count consecutive build days from most recent
        for d in daily_oi:
            if d["net_oi_change"] > 0:
                result["consecutive_build_days"] += 1
            else:
                break

        return text_result(result)

    elif name == "cumulative_premium_flow":
        symbol = arguments["symbol"].upper()
        days = int(arguments.get("days", 90))
        dates = list_available_dates("screener")[:days]
        if not dates:
            return text_result(f"No screener data available for {symbol}")
        cum_bull = 0.0
        cum_bear = 0.0
        dates_covered = []
        for d in dates:
            try:
                df = load_data("screener", d)
                row = df[df["ticker"] == symbol]
                if row.empty:
                    continue
                r = row.iloc[0]
                cum_bull += float(r.get("bullish_premium", 0) or 0)
                cum_bear += float(r.get("bearish_premium", 0) or 0)
                dates_covered.append(d)
            except Exception:
                continue
        if not dates_covered:
            return text_result(f"No data found for {symbol} in last {days} sessions")
        net = cum_bull - cum_bear
        if abs(net) < 0.05 * (cum_bull + cum_bear + 1):
            trend = "MIXED"
        else:
            trend = "BULLISH" if net > 0 else "BEARISH"
        return text_result({
            "symbol": symbol,
            "days": days,
            "dates_covered": dates_covered,
            "cumulative_bullish": round(cum_bull, 2),
            "cumulative_bearish": round(cum_bear, 2),
            "net_flow": round(net, 2),
            "trend_direction": trend,
        })

    elif name == "pc_ratio_zscore":
        symbol = arguments["symbol"].upper()
        lookback = int(arguments.get("lookback_days", 20))
        dates = list_available_dates("screener")[:lookback + 1]
        if len(dates) < 5:
            return text_result(f"Need at least 5 sessions of data; have {len(dates)}")
        ratios = []
        for d in dates:
            try:
                df = load_data("screener", d)
                row = df[df["ticker"] == symbol]
                if row.empty:
                    continue
                pc = row.iloc[0].get("put_call_ratio")
                if pc is None:
                    continue
                ratios.append(float(pc))
            except Exception:
                continue
        if len(ratios) < 5:
            return text_result(f"Need at least 5 P/C readings for {symbol}; have {len(ratios)}")
        current = ratios[0]
        history = ratios[1:]
        import statistics
        mean = statistics.mean(history)
        std = statistics.stdev(history) if len(history) > 1 else 0.0
        z = (current - mean) / std if std > 0 else 0.0
        if z > 2:
            extreme = "BEARISH_EXTREME"
        elif z < -2:
            extreme = "BULLISH_EXTREME"
        else:
            extreme = "NORMAL"
        return text_result({
            "symbol": symbol,
            "current_pc_ratio": round(current, 4),
            "mean_pc_ratio": round(mean, 4),
            "std_pc_ratio": round(std, 4),
            "zscore": round(z, 3),
            "extreme": extreme,
            "lookback_days": len(history),
        })

    elif name == "iv_percentile_zscore":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper()
        lookback_days = int(arguments.get("lookback_days", 252))
        if lookback_days < 5:
            raise ValueError(
                f"lookback_days must be >= 5 to compute a meaningful distribution; got {lookback_days}"
            )

        dates = iter_dates("screener", n=lookback_days + 1)
        if len(dates) < 5:
            return text_result({
                "symbol": symbol,
                "note": f"Need at least 5 sessions of screener data; have {len(dates)}",
            })

        ivs = []
        for d in dates:
            try:
                df = load_data("screener", d, usecols=["ticker", "iv30d"])
                row = df[df["ticker"] == symbol]
                if row.empty:
                    continue
                iv = row.iloc[0].get("iv30d")
                if iv is None or pd.isna(iv):
                    continue
                ivs.append(float(iv))
            except Exception:
                continue

        if len(ivs) < 5:
            return text_result({
                "symbol": symbol,
                "note": f"Need at least 5 IV30d readings; have {len(ivs)}",
            })

        current = ivs[0]
        history = ivs[1:]
        pct = percentile_rank(current, history)
        z = zscore(current, history)

        if pct is not None and pct >= 80.0:
            regime = "HIGH_IV"
        elif pct is not None and pct <= 20.0:
            regime = "LOW_IV"
        else:
            regime = "NORMAL"

        return text_result({
            "symbol": symbol,
            "current_iv30d": round(current, 4),
            "iv_percentile": round(pct, 2) if pct is not None else None,
            "iv_zscore": round(z, 3) if z is not None else None,
            "regime": regime,
            "dates_used": len(history),
            "lookback_days": lookback_days,
        })

    elif name == "gex_time_series":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper()
        days = int(arguments.get("days", 30))
        dte_max_raw = arguments.get("dte_max", 45)
        dte_max = int(dte_max_raw) if dte_max_raw is not None else None
        if days < 1 or days > 500:
            raise ValueError(f"days must be 1-500, got {days}")
        if dte_max is not None and (dte_max < 1 or dte_max > 1500):
            raise ValueError(f"dte_max must be 1-1500 or null, got {dte_max}")

        file_dates = iter_dates("options", n=days)
        if not file_dates:
            return text_result({
                "symbol": symbol,
                "note": "No options Parquet files available",
            })

        # Walk dates oldest → newest so regime-flip detection compares chronological neighbours
        file_dates_chrono = list(reversed(file_dates))
        trajectory: list[dict] = []
        for fd in file_dates_chrono:
            day = _compute_gex_for_date(symbol, fd, dte_max)
            if day is None:
                continue
            trajectory.append({"date": fd, **day})

        if not trajectory:
            return text_result({
                "symbol": symbol,
                "note": f"No GEX-eligible options data for {symbol} in last {days} sessions",
            })

        # Detect regime-flip dates: spot crosses ZGL between consecutive sessions.
        # `zgl_delta` lets callers distinguish a spot-driven flip (small delta)
        # from a ZGL-driven flip (large delta — large dealer-position shift).
        regime_flip_dates: list[dict] = []
        for prev, curr in zip(trajectory, trajectory[1:]):
            zgl_prev, zgl_curr = prev["zero_gamma_level"], curr["zero_gamma_level"]
            if zgl_prev is None or zgl_curr is None:
                continue
            prev_above = prev["spot"] > zgl_prev
            curr_above = curr["spot"] > zgl_curr
            if prev_above != curr_above:
                regime_flip_dates.append({
                    "date": curr["date"],
                    "from_regime": prev["regime"],
                    "to_regime": curr["regime"],
                    "spot": curr["spot"],
                    "zero_gamma_level": zgl_curr,
                    "zgl_delta": round(zgl_curr - zgl_prev, 2),
                })

        return text_result({
            "symbol": symbol,
            "days_analyzed": len(trajectory),
            "dte_max": dte_max,
            "trajectory": trajectory,
            "regime_flip_dates": regime_flip_dates,
            "note": (
                "Regime flips identify dates where spot crossed the zero-gamma level "
                "(POSITIVE↔NEGATIVE). Empirically precedes realised-vol expansion."
            ),
        })

    elif name == "volatility_risk_premium":
        if not arguments.get("symbol"):
            raise ValueError("symbol is required")
        symbol = arguments["symbol"].upper()
        date_arg = arguments.get("date")
        realised_window = int(arguments.get("realised_window_days", 30))
        if realised_window < 5 or realised_window > 252:
            raise ValueError(
                f"realised_window_days must be 5-252, got {realised_window}"
            )

        # Latest IV30d for the symbol
        try:
            screener = load_data("screener", date_arg, usecols=["ticker", "iv30d"])
        except FileNotFoundError:
            return text_result({"error": "No screener data available"})
        row = screener[screener["ticker"] == symbol]
        if row.empty:
            return text_result({"symbol": symbol, "note": f"No screener entry for {symbol}"})
        iv30d_raw = row.iloc[0].get("iv30d")
        if iv30d_raw is None or pd.isna(iv30d_raw):
            return text_result({"symbol": symbol, "note": "iv30d unavailable"})
        iv30d = float(iv30d_raw)
        # UW occasionally stores IV as percentage (e.g. 30.0 instead of 0.30).
        # Anything > 5.0 (>500% annualised) cannot be a decimal in practice.
        if iv30d > 5.0:
            iv30d = iv30d / 100.0

        # Realised vol from yfinance close-to-close. Use ~1.5× buffer over the
        # requested trading-day window to absorb weekends + holiday clusters.
        try:
            t = Ticker(symbol)
            calendar_buffer = max(realised_window + 14, int(realised_window * 1.5))
            hist = t.history(period=f"{calendar_buffer}d")
            if hist is None or hist.empty:
                return text_result({
                    "symbol": symbol,
                    "error": "yfinance returned no price history",
                })
            realised = realised_volatility(hist["Close"].tail(realised_window + 1))
        except Exception as e:
            return text_result({"symbol": symbol, "error": f"realised vol unavailable: {e}"})

        if realised is None:
            return text_result({"symbol": symbol, "error": "insufficient price history"})

        vrp = iv30d - realised
        # +/- 5% threshold for regime classification
        if vrp > 0.05:
            regime = "PREMIUM_SELLING"
            interpretation = "Options pricing more vol than realised — favour premium selling."
        elif vrp < -0.05:
            regime = "PREMIUM_BUYING"
            interpretation = "Vol cheap vs realised — favour premium buying."
        else:
            regime = "FAIR"
            interpretation = "IV close to realised — no clear edge from VRP alone."

        return text_result({
            "symbol": symbol,
            "iv30d": round(iv30d, 4),
            "realised_vol": round(realised, 4),
            "realised_window_days": realised_window,
            "vrp": round(vrp, 4),
            "regime": regime,
            "interpretation": interpretation,
        })

    elif name == "signal_backtest":
        signal_type = arguments["signal_type"]
        lookback_days = arguments.get("lookback_days", 5)
        top_n = arguments.get("top_n", 20)
        dates = list_available_dates("screener")

        if len(dates) < 1:
            return text_result({
                "signal_type": signal_type,
                "total_signals": 0,
                "note": "No screener data available. Run convert.py to populate Parquet files.",
            })

        signals_found = []

        for date in dates:
            try:
                df = load_data("screener", date)

                if signal_type == "bullish_flow":
                    df["net_flow"] = df["bullish_premium"].fillna(0) - df["bearish_premium"].fillna(0)
                    flagged = df[df["net_flow"] > 0].sort_values("net_flow", ascending=False).head(5)
                elif signal_type == "bearish_flow":
                    df["net_flow"] = df["bullish_premium"].fillna(0) - df["bearish_premium"].fillna(0)
                    flagged = df[df["net_flow"] < 0].sort_values("net_flow", ascending=True).head(5)
                elif signal_type == "high_iv_rank":
                    flagged = df[df["iv_rank"].astype(float) > 80].sort_values("iv_rank", ascending=False).head(5)
                elif signal_type == "volume_spike":
                    df["total_vol"] = df["call_volume"].fillna(0) + df["put_volume"].fillna(0)
                    df["avg_vol"] = df["avg_30_day_call_volume"].fillna(0) + df["avg_30_day_put_volume"].fillna(0)
                    df = df[df["avg_vol"] > 0]
                    df["vol_ratio"] = df["total_vol"] / df["avg_vol"]
                    flagged = df[df["vol_ratio"] > 3].sort_values("vol_ratio", ascending=False).head(5)
                elif signal_type == "dark_pool_accumulation":
                    try:
                        dp = load_data("darkpool", date)
                        dp["premium"] = dp["premium"].astype(float)
                        dp_agg = dp.groupby("ticker").agg(total_premium=("premium", "sum")).reset_index()
                        flagged_tickers = dp_agg.sort_values("total_premium", ascending=False).head(5)["ticker"].tolist()
                        flagged = df[df["ticker"].isin(flagged_tickers)]
                    except Exception:
                        continue
                else:
                    continue

                for _, row in flagged.iterrows():
                    signals_found.append({
                        "date": date,
                        "ticker": row["ticker"],
                        "close_on_signal": row.get("close"),
                    })
            except Exception:
                continue

        # Check price action after each signal using Yahoo Finance.
        # `lookback_days` = TRADING days. Calendar buffer of +14 covers any
        # plausible holiday cluster within reasonable lookback windows.
        # Audit §1.4 fix for the Friday-before-holiday lookup bug.
        results = []
        truncated_count = 0
        seen = set()
        calendar_buffer = lookback_days + 14
        for sig in signals_found[:top_n]:
            key = f"{sig['ticker']}_{sig['date']}"
            if key in seen:
                continue
            seen.add(key)
            try:
                t = Ticker(sig["ticker"])
                start = pd.to_datetime(sig["date"]) + pd.Timedelta(days=1)
                end = start + pd.Timedelta(days=calendar_buffer)
                hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
                if hist is None or hist.empty or len(hist) < 2:
                    continue
                # Trading-day index: bar 0 is the first session AFTER the signal,
                # so lookback_days=N means we want bar (N-1) but bounded by len-1.
                bar_idx = min(lookback_days - 1, len(hist) - 1)
                if bar_idx + 1 < lookback_days:
                    truncated_count += 1
                close_after = float(hist["Close"].iloc[bar_idx])
                # Skip signals where the screener close is missing — falling back
                # to the next session's bar would silently bias pct_change toward
                # zero by removing the overnight gap. Better to drop the row.
                close_on_signal_raw = sig.get("close_on_signal")
                if close_on_signal_raw is None or pd.isna(close_on_signal_raw):
                    continue
                close_on_signal = float(close_on_signal_raw)
                pct_change = ((close_after - close_on_signal) / close_on_signal) * 100
                results.append({
                    "ticker": sig["ticker"],
                    "signal_date": sig["date"],
                    "price_on_signal": round(close_on_signal, 2),
                    f"price_after_{lookback_days}d": round(close_after, 2),
                    "pct_change": round(pct_change, 2),
                    "direction": "up" if pct_change > 0 else "down",
                })
            except Exception:
                continue

        if not results:
            return text_result({
                "signal_type": signal_type,
                "total_signals": 0,
                "note": "No backtest results. Need more historical data or valid signals.",
            })

        avg_move = sum(r["pct_change"] for r in results) / len(results)

        # Metric semantics depend on signal type:
        #   - directional (bullish_flow, bearish_flow, dark_pool_accumulation)
        #     → win_rate (move agrees with thesis)
        #   - non-directional (high_iv_rank, volume_spike) → vol_realisation_rate
        #     (any move > 2%; the signal has no directional thesis to win on).
        #     Audit §1.4 fix.
        non_directional = {"high_iv_rank", "volume_spike"}
        summary: dict = {
            "signal_type": signal_type,
            "total_signals": len(results),
            "avg_move_pct": round(avg_move, 2),
            "lookback_days": lookback_days,
            "truncated_signals": truncated_count,
            "results": results,
        }
        if len(results) < 5:
            summary["low_sample_warning"] = (
                f"Only {len(results)} forward-return observations — treat as anecdote, not statistic."
            )
        if signal_type in non_directional:
            vol_realised = sum(1 for r in results if abs(r["pct_change"]) > 2)
            summary["vol_realisation_rate"] = (
                f"{(vol_realised / len(results) * 100):.1f}%"
            )
            summary["methodology_notes"] = (
                f"{signal_type} is non-directional — vol_realisation_rate measures "
                "the fraction of signals where realised |move| > 2% within the lookback "
                "window. Lookback is in TRADING days; selection is positional within "
                "the yfinance bar series."
            )
        else:
            if signal_type == "bearish_flow":
                wins = sum(1 for r in results if r["pct_change"] < 0)
            else:
                # bullish_flow / dark_pool_accumulation
                wins = sum(1 for r in results if r["pct_change"] > 0)
            summary["win_rate"] = f"{(wins / len(results) * 100):.1f}%"
            summary["methodology_notes"] = (
                "win_rate = fraction of signals where forward move agrees with the "
                "signal's direction. Lookback is in TRADING days; selection is positional "
                "within the yfinance bar series. In-sample backtest — not a robust live edge."
            )
        return text_result(summary)

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
