"""MCP Server: Risk/correlation analysis and market regime detection."""

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

from shared.data_loader import load_data
from shared.dp_classification import classify_dp_side
from shared.server_utils import text_result

server = Server("uw-risk")


def compute_market_regime(date: str | None) -> dict:
    """Classify current market conditions using SPY trend, VIX, breadth, and sector rotation."""
    spy = Ticker("SPY")
    spy_hist = spy.history(period="90d")

    vix = Ticker("^VIX")
    vix_hist = vix.history(period="30d")

    result: dict = {"date": date or "latest"}

    if not spy_hist.empty:
        current = float(spy_hist["Close"].iloc[-1])
        sma_20 = float(spy_hist["Close"].tail(20).mean())
        sma_50 = float(spy_hist["Close"].tail(50).mean())
        high_52w = float(spy_hist["High"].max())
        change_30d = ((current - float(spy_hist["Close"].iloc[-min(22, len(spy_hist))])) / float(spy_hist["Close"].iloc[-min(22, len(spy_hist))])) * 100
        pct_from_high = ((current - high_52w) / high_52w) * 100

        result["spy"] = {
            "current": round(current, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "above_20sma": current > sma_20,
            "above_50sma": current > sma_50,
            "pct_from_90d_high": round(pct_from_high, 2),
            "change_30d_pct": round(change_30d, 2),
        }

        if current > sma_20 > sma_50:
            trend = "UPTREND"
        elif current < sma_20 < sma_50:
            trend = "DOWNTREND"
        elif current > sma_50 and current < sma_20:
            trend = "PULLBACK_IN_UPTREND"
        elif current < sma_50 and current > sma_20:
            trend = "BOUNCE_IN_DOWNTREND"
        else:
            trend = "CHOPPY"
        result["trend"] = trend

    if not vix_hist.empty:
        vix_current = float(vix_hist["Close"].iloc[-1])
        vix_avg = float(vix_hist["Close"].mean())
        result["vix"] = {
            "current": round(vix_current, 2),
            "30d_avg": round(vix_avg, 2),
            "level": "LOW" if vix_current < 15 else ("ELEVATED" if vix_current < 25 else ("HIGH" if vix_current < 35 else "EXTREME")),
        }

    try:
        screener = load_data("screener", date)
        total = len(screener)
        bullish = len(screener[screener["bullish_premium"].fillna(0) > screener["bearish_premium"].fillna(0)])
        result["market_breadth"] = {
            "tickers_with_options": total,
            "bullish_flow_tickers": bullish,
            "bearish_flow_tickers": total - bullish,
            "bullish_pct": round(bullish / total * 100, 1) if total > 0 else 0,
        }
    except Exception:
        pass

    try:
        screener = load_data("screener", date)
        screener["net"] = screener["bullish_premium"].fillna(0) - screener["bearish_premium"].fillna(0)
        sector_flow = screener.groupby("sector")["net"].sum().sort_values(ascending=False)
        result["sector_rotation"] = {
            "money_flowing_in": sector_flow.head(3).to_dict(),
            "money_flowing_out": sector_flow.tail(3).to_dict(),
        }
    except Exception:
        pass

    regime_signals = []
    if result.get("trend") in ["UPTREND"]:
        regime_signals.append("bullish_trend")
    elif result.get("trend") in ["DOWNTREND"]:
        regime_signals.append("bearish_trend")

    vix_level = result.get("vix", {}).get("level", "")
    if vix_level in ["LOW"]:
        regime_signals.append("low_fear")
    elif vix_level in ["HIGH", "EXTREME"]:
        regime_signals.append("high_fear")

    breadth = result.get("market_breadth", {}).get("bullish_pct", 50)
    if breadth > 60:
        regime_signals.append("broad_bullish")
    elif breadth < 40:
        regime_signals.append("broad_bearish")

    if "bullish_trend" in regime_signals and "low_fear" in regime_signals:
        result["regime"] = "RISK-ON — Favor bullish strategies, trend-following"
    elif "bearish_trend" in regime_signals and "high_fear" in regime_signals:
        result["regime"] = "RISK-OFF — Favor defensive strategies, hedges, or cash"
    elif "bearish_trend" in regime_signals and "low_fear" in regime_signals:
        result["regime"] = "COMPLACENT — Market declining but VIX low, potential for volatility spike"
    elif "bullish_trend" in regime_signals and "high_fear" in regime_signals:
        result["regime"] = "WALL OF WORRY — Market rising despite fear, often strongest rallies"
    else:
        result["regime"] = "TRANSITIONAL — Mixed signals, reduce position size, wait for clarity"

    result["trading_guidance"] = {
        "RISK-ON": "Full position sizes. Favor long calls, bull put spreads. Sell puts on dips.",
        "RISK-OFF": "Reduce size. Favor long puts, bear call spreads. Cash is a position.",
        "COMPLACENT": "Buy cheap protection (VIX calls, SPY puts). Tighten stops.",
        "WALL OF WORRY": "Stay long but hedge. Bull call spreads over naked calls.",
        "TRANSITIONAL": "Half position sizes. Favor defined-risk strategies. Iron condors in range.",
    }.get(result["regime"].split(" — ")[0], "")

    return result


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="portfolio_correlation",
            description=(
                "Check sector/industry concentration and correlation of a set of tickers. "
                "Warns if your trade ideas are all in one basket. Uses Yahoo Finance for "
                "sector/industry data and calculates price correlations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ticker symbols to check",
                    },
                    "lookback_days": {"type": "integer", "description": "Days for correlation calc (default: 30)", "default": 30},
                },
                "required": ["symbols"],
            },
        ),
        Tool(
            name="market_regime",
            description=(
                "Classify current market conditions using SPY trend, VIX level, "
                "market breadth, and sector rotation. Helps you weight signals — "
                "bullish sweeps mean less in a bear market."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    if name == "portfolio_correlation":
        symbols = [s.upper() for s in arguments["symbols"]]
        lookback = arguments.get("lookback_days", 30)

        # Get sector/industry info
        ticker_info = []
        price_data = {}
        for sym in symbols:
            try:
                t = Ticker(sym)
                info = t.info
                ticker_info.append({
                    "symbol": sym,
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "market_cap": info.get("marketCap"),
                    "beta": info.get("beta"),
                })
                hist = t.history(period=f"{lookback}d")
                if not hist.empty:
                    price_data[sym] = hist["Close"]
            except Exception:
                ticker_info.append({"symbol": sym, "sector": "Unknown", "industry": "Unknown"})

        # Sector concentration
        sectors = {}
        for ti in ticker_info:
            s = ti["sector"]
            sectors[s] = sectors.get(s, 0) + 1

        # Correlation matrix
        correlations = []
        if len(price_data) >= 2:
            df_prices = pd.DataFrame(price_data)
            returns = df_prices.pct_change().dropna()
            if len(returns) >= 5:
                corr = returns.corr()
                # Find high correlations
                for i, s1 in enumerate(corr.columns):
                    for j, s2 in enumerate(corr.columns):
                        if i < j:
                            c = round(float(corr.loc[s1, s2]), 3)
                            if abs(c) >= 0.5:
                                correlations.append({
                                    "pair": f"{s1}/{s2}",
                                    "correlation": c,
                                    "warning": "HIGH" if abs(c) >= 0.8 else "MODERATE",
                                })
                correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Risk assessment
        warnings = []
        max_sector_pct = max(sectors.values()) / len(symbols) * 100 if symbols else 0
        if max_sector_pct >= 60:
            top_sector = max(sectors, key=sectors.get)
            warnings.append(f"CONCENTRATION: {max_sector_pct:.0f}% of tickers in {top_sector}")
        if any(c["correlation"] >= 0.8 for c in correlations):
            warnings.append("HIGH CORRELATION: Some pairs move nearly in lockstep — not truly diversified")
        high_beta = [ti for ti in ticker_info if ti.get("beta") and ti["beta"] > 1.5]
        if len(high_beta) > len(symbols) / 2:
            warnings.append(f"HIGH BETA: {len(high_beta)}/{len(symbols)} tickers have beta > 1.5")

        result = {
            "tickers_analyzed": len(symbols),
            "sector_breakdown": sectors,
            "sector_concentration": f"{max_sector_pct:.0f}% in top sector",
            "ticker_details": ticker_info,
            "high_correlations": correlations[:10],
            "warnings": warnings if warnings else ["Portfolio looks reasonably diversified"],
        }
        return text_result(result)

    elif name == "market_regime":
        return text_result(compute_market_regime(arguments.get("date")))

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
