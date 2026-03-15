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

from shared.data_loader import load_data, list_available_dates
from shared.server_utils import text_result, df_to_result

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

    elif name == "signal_backtest":
        signal_type = arguments["signal_type"]
        lookback_days = arguments.get("lookback_days", 5)
        top_n = arguments.get("top_n", 20)
        dates = list_available_dates("screener")

        if len(dates) < 2:
            return text_result("Need at least 2 days of data for backtesting. Accumulate more daily downloads.")

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

        # Now check price action after each signal using Yahoo Finance
        results = []
        seen = set()
        for sig in signals_found[:top_n]:
            key = f"{sig['ticker']}_{sig['date']}"
            if key in seen:
                continue
            seen.add(key)
            try:
                t = Ticker(sig["ticker"])
                start = pd.to_datetime(sig["date"]) + pd.Timedelta(days=1)
                end = start + pd.Timedelta(days=lookback_days + 5)  # extra buffer for weekends
                hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
                if hist.empty or len(hist) < 2:
                    continue
                close_after = float(hist["Close"].iloc[min(lookback_days - 1, len(hist) - 1)])
                close_on_signal = float(sig["close_on_signal"]) if sig["close_on_signal"] else float(hist["Close"].iloc[0])
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
            return text_result("No backtest results. Need more historical data or valid signals.")

        # Summary stats
        wins = sum(1 for r in results if (signal_type in ["bullish_flow", "volume_spike", "dark_pool_accumulation"] and r["pct_change"] > 0) or (signal_type == "bearish_flow" and r["pct_change"] < 0) or (signal_type == "high_iv_rank" and abs(r["pct_change"]) > 2))
        avg_move = sum(r["pct_change"] for r in results) / len(results)

        summary = {
            "signal_type": signal_type,
            "total_signals": len(results),
            "win_rate": f"{(wins / len(results) * 100):.1f}%",
            "avg_move_pct": round(avg_move, 2),
            "lookback_days": lookback_days,
            "results": results,
        }
        return text_result(summary)

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
