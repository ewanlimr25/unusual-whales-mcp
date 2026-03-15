"""MCP Server: Options strategy suggestions based on signals."""

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
from shared.server_utils import text_result

server = Server("uw-strategy")


def _analyze_signals(symbol: str, date=None) -> dict:
    """Gather all available signals for a ticker."""
    signals = {"symbol": symbol, "factors": []}

    # Screener data
    try:
        screener = load_data("screener", date)
        row = screener[screener["ticker"] == symbol]
        if not row.empty:
            r = row.iloc[0]
            iv_rank = float(r.get("iv_rank", 0) or 0)
            pcr = float(r.get("put_call_ratio", 0) or 0)
            bull = float(r.get("bullish_premium", 0) or 0)
            bear = float(r.get("bearish_premium", 0) or 0)
            net_flow = bull - bear
            total_vol = float(r.get("call_volume", 0) or 0) + float(r.get("put_volume", 0) or 0)
            avg_vol = float(r.get("avg_30_day_call_volume", 0) or 0) + float(r.get("avg_30_day_put_volume", 0) or 0)
            vol_ratio = total_vol / avg_vol if avg_vol > 0 else 1
            implied_move = r.get("implied_move_perc")
            next_er = r.get("next_earnings_date")

            signals["close"] = r.get("close")
            signals["iv_rank"] = iv_rank
            signals["put_call_ratio"] = pcr
            signals["net_flow"] = net_flow
            signals["flow_direction"] = "bullish" if net_flow > 0 else "bearish"
            signals["volume_ratio"] = round(vol_ratio, 2)
            signals["implied_move_perc"] = implied_move
            signals["next_earnings"] = str(next_er) if pd.notna(next_er) else None

            if iv_rank >= 70:
                signals["factors"].append("HIGH_IV")
            elif iv_rank <= 20:
                signals["factors"].append("LOW_IV")
            if net_flow > 0:
                signals["factors"].append("BULLISH_FLOW")
            else:
                signals["factors"].append("BEARISH_FLOW")
            if vol_ratio >= 2:
                signals["factors"].append("VOLUME_SPIKE")
            if pcr >= 2:
                signals["factors"].append("HIGH_PUT_CALL")
            elif pcr <= 0.5:
                signals["factors"].append("LOW_PUT_CALL")
    except Exception:
        pass

    # Dark pool
    try:
        dp = load_data("darkpool", date)
        dp_t = dp[dp["ticker"] == symbol]
        if not dp_t.empty:
            dp_t = dp_t.copy()
            dp_t["premium"] = dp_t["premium"].astype(float)
            dp_t["price"] = dp_t["price"].astype(float)
            dp_t["nbbo_bid"] = dp_t["nbbo_bid"].astype(float)
            dp_t["nbbo_ask"] = dp_t["nbbo_ask"].astype(float)
            dp_t["mid"] = (dp_t["nbbo_bid"] + dp_t["nbbo_ask"]) / 2
            buy_vol = int(dp_t[dp_t["price"] >= dp_t["mid"]]["size"].astype(int).sum())
            sell_vol = int(dp_t[dp_t["price"] < dp_t["mid"]]["size"].astype(int).sum())
            signals["dp_buy_volume"] = buy_vol
            signals["dp_sell_volume"] = sell_vol
            if buy_vol > sell_vol * 1.5:
                signals["factors"].append("DP_ACCUMULATION")
            elif sell_vol > buy_vol * 1.5:
                signals["factors"].append("DP_DISTRIBUTION")
    except Exception:
        pass

    # OI changes
    try:
        oi = load_data("oi", date)
        oi_t = oi[oi["underlying_symbol"] == symbol]
        if not oi_t.empty:
            oi_t["oi_diff_plain"] = oi_t["oi_diff_plain"].astype(float)
            net_oi = float(oi_t["oi_diff_plain"].sum())
            signals["net_oi_change"] = net_oi
            if net_oi > 5000:
                signals["factors"].append("OI_BUILDING")
            elif net_oi < -5000:
                signals["factors"].append("OI_UNWINDING")
    except Exception:
        pass

    # Yahoo data
    try:
        t = Ticker(symbol)
        info = t.info
        signals["market_cap"] = info.get("marketCap")
        signals["sector"] = info.get("sector")
        signals["short_percent"] = info.get("shortPercentOfFloat")
        if signals.get("short_percent") and signals["short_percent"] > 0.15:
            signals["factors"].append("HIGH_SHORT_INTEREST")
    except Exception:
        pass

    return signals


def _suggest_strategies(signals: dict) -> list[dict]:
    """Generate strategy suggestions based on signal factors."""
    factors = set(signals.get("factors", []))
    iv_rank = signals.get("iv_rank", 50)
    strategies = []

    # === HIGH IV STRATEGIES (sell premium) ===
    if "HIGH_IV" in factors:
        if "BULLISH_FLOW" in factors or "DP_ACCUMULATION" in factors:
            strategies.append({
                "strategy": "Bull Put Spread (Credit Spread)",
                "reasoning": "IV is elevated (good premiums for selling) and flow is bullish. Sell a put spread below current price to collect premium with defined risk.",
                "risk": "Limited to spread width minus premium received",
                "ideal_if": "You're moderately bullish and want to profit from IV contraction + upside",
                "setup": "Sell OTM put, buy further OTM put. Target 30-45 DTE. Look for 1/3 width credit.",
            })
        elif "BEARISH_FLOW" in factors or "DP_DISTRIBUTION" in factors:
            strategies.append({
                "strategy": "Bear Call Spread (Credit Spread)",
                "reasoning": "IV is elevated and flow is bearish. Sell a call spread above current price.",
                "risk": "Limited to spread width minus premium received",
                "ideal_if": "You're moderately bearish and want to sell expensive premium",
                "setup": "Sell OTM call, buy further OTM call. Target 30-45 DTE.",
            })
        else:
            strategies.append({
                "strategy": "Iron Condor",
                "reasoning": "IV is high but no strong directional signal. Sell premium on both sides and profit from IV contraction.",
                "risk": "Limited to wider spread width minus total premium",
                "ideal_if": "You expect the stock to stay range-bound or IV to contract",
                "setup": "Sell OTM put spread + OTM call spread. Target 30-45 DTE, 16-delta wings.",
            })

        if signals.get("next_earnings"):
            strategies.append({
                "strategy": "Pre-Earnings Iron Condor / Short Strangle",
                "reasoning": f"Earnings approaching ({signals['next_earnings']}) with elevated IV. IV will likely crush after earnings.",
                "risk": "High if stock gaps beyond your strikes. Use defined risk (iron condor) to cap losses.",
                "ideal_if": "You believe implied move overstates the actual move",
                "setup": f"Set strikes outside implied move ({signals.get('implied_move_perc', 'N/A')}). Close immediately after earnings.",
            })

    # === LOW IV STRATEGIES (buy premium) ===
    if "LOW_IV" in factors:
        if "BULLISH_FLOW" in factors:
            strategies.append({
                "strategy": "Long Call or Call Debit Spread",
                "reasoning": "Options are cheap (low IV rank) and smart money is bullish. Good time to buy calls.",
                "risk": "Limited to premium paid",
                "ideal_if": "You're bullish and want leveraged upside with cheap options",
                "setup": "Buy ATM or slightly OTM call. 45-60 DTE for time. Or buy call spread to reduce cost.",
            })
        elif "BEARISH_FLOW" in factors:
            strategies.append({
                "strategy": "Long Put or Put Debit Spread",
                "reasoning": "Options are cheap and flow is bearish. Buy puts while IV is depressed.",
                "risk": "Limited to premium paid",
                "ideal_if": "You're bearish and want cheap downside protection or speculation",
                "setup": "Buy ATM or slightly OTM put. 45-60 DTE.",
            })
        if "VOLUME_SPIKE" in factors or "OI_BUILDING" in factors:
            strategies.append({
                "strategy": "Long Straddle / Strangle",
                "reasoning": "IV is low but unusual activity suggests a big move coming. Buy both sides cheap.",
                "risk": "Limited to premium paid, but needs a large move to profit",
                "ideal_if": "You expect a big move but aren't sure of direction",
                "setup": "Buy ATM straddle or OTM strangle. 30-45 DTE. Need stock to move beyond implied range.",
            })

    # === DIRECTIONAL WITH CONFIRMATION ===
    if "BULLISH_FLOW" in factors and "DP_ACCUMULATION" in factors:
        strategies.append({
            "strategy": "Aggressive Long Calls (Multi-Signal Bullish)",
            "reasoning": "Both options flow AND dark pool indicate bullish positioning. Multiple signals confirm direction.",
            "risk": "Limited to premium paid, but higher confidence trade",
            "ideal_if": "You want strong directional exposure with signal confluence",
            "setup": "Buy ITM or ATM calls for delta exposure. 30-60 DTE. Size modestly.",
        })
    if "BEARISH_FLOW" in factors and "DP_DISTRIBUTION" in factors:
        strategies.append({
            "strategy": "Aggressive Long Puts (Multi-Signal Bearish)",
            "reasoning": "Both options flow AND dark pool show bearish signals. Strong confluence.",
            "risk": "Limited to premium paid",
            "ideal_if": "You want directional downside exposure with high conviction",
            "setup": "Buy ITM or ATM puts. 30-60 DTE.",
        })

    # === SQUEEZE SETUP ===
    if "HIGH_SHORT_INTEREST" in factors and "BULLISH_FLOW" in factors:
        strategies.append({
            "strategy": "Short Squeeze Play (Long Calls + Shares)",
            "reasoning": f"High short interest ({signals.get('short_percent', 'N/A')}) combined with bullish flow. Potential squeeze setup.",
            "risk": "High volatility expected. Size small.",
            "ideal_if": "You want to position for a potential short squeeze",
            "setup": "Buy slightly OTM calls 30-45 DTE. Or buy shares for no time decay risk.",
        })

    # === OI BUILDING ===
    if "OI_BUILDING" in factors and "VOLUME_SPIKE" in factors:
        strategies.append({
            "strategy": "Follow the Smart Money (Directional)",
            "reasoning": "New positions being opened with elevated volume. Fresh conviction entering.",
            "risk": "Depends on strategy chosen",
            "ideal_if": "You want to piggyback on institutional positioning",
            "setup": "Follow the direction of the OI build. Check if calls or puts are building to determine direction.",
        })

    if not strategies:
        strategies.append({
            "strategy": "No Clear Edge \u2014 Stay Flat",
            "reasoning": "No strong signal confluence detected. The best trade is sometimes no trade.",
            "risk": "N/A",
            "ideal_if": "Signals are mixed or weak",
            "setup": "Wait for clearer signals or look at other tickers.",
        })

    return strategies


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="suggest_strategy",
            description=(
                "Analyze all available signals for a ticker and suggest specific options "
                "strategies. Combines UW flow data, dark pool, OI changes, IV rank, and "
                "Yahoo Finance fundamentals to generate actionable trade ideas."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="batch_strategy_scan",
            description=(
                "Run strategy analysis on multiple tickers at once. "
                "Great for scanning your watchlist or a sector for trade ideas."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ticker symbols to analyze",
                    },
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                },
                "required": ["symbols"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    if name == "suggest_strategy":
        symbol = arguments["symbol"].upper()
        date = arguments.get("date")
        signals = _analyze_signals(symbol, date)
        strategies = _suggest_strategies(signals)
        result = {
            "symbol": symbol,
            "signals_detected": signals["factors"],
            "signal_data": {k: v for k, v in signals.items() if k not in ["factors", "symbol"]},
            "suggested_strategies": strategies,
        }
        return text_result(result)

    elif name == "batch_strategy_scan":
        symbols = [s.upper() for s in arguments["symbols"]]
        date = arguments.get("date")
        results = []
        for symbol in symbols:
            signals = _analyze_signals(symbol, date)
            strategies = _suggest_strategies(signals)
            # Only include tickers with actionable strategies
            has_edge = any(s["strategy"] != "No Clear Edge \u2014 Stay Flat" for s in strategies)
            results.append({
                "symbol": symbol,
                "factors": signals["factors"],
                "has_edge": has_edge,
                "top_strategy": strategies[0]["strategy"] if strategies else "None",
                "top_reasoning": strategies[0]["reasoning"] if strategies else "",
                "iv_rank": signals.get("iv_rank"),
                "flow_direction": signals.get("flow_direction"),
                "close": signals.get("close"),
            })
        # Sort: tickers with edge first, then by number of factors
        results.sort(key=lambda x: (not x["has_edge"], -len(x["factors"])))
        return text_result(results)

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
