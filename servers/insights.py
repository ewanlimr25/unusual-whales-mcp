"""MCP Server: Cross-dataset analysis combining Unusual Whales data with Yahoo Finance."""

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

server = Server("uw-insights")


def _get_yahoo_data(symbol: str) -> dict:
    """Fetch key Yahoo Finance data for a symbol."""
    t = Ticker(symbol)
    info = t.info
    return {
        "current_price": info.get("regularMarketPrice", info.get("currentPrice")),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "avg_volume": info.get("averageVolume"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "short_ratio": info.get("shortRatio"),
        "short_percent": info.get("shortPercentOfFloat"),
    }


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="stock_deep_dive",
            description=(
                "Comprehensive analysis of a single ticker combining Yahoo Finance "
                "fundamentals with Unusual Whales options flow, dark pool, and screener data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "date": {"type": "string", "description": "Date for UW data (YYYY-MM-DD). Defaults to latest."},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="earnings_play_analyzer",
            description=(
                "Find pre-earnings setups: combines Yahoo earnings dates with UW IV rank "
                "and OI changes to identify unusual pre-earnings positioning."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "days_until_earnings": {"type": "integer", "description": "Max days to earnings (default: 14)", "default": 14},
                    "min_iv_rank": {"type": "number", "description": "Min IV rank (default: 40)", "default": 40},
                    "top_n": {"type": "integer", "description": "Number of results (default: 10)", "default": 10},
                },
                "required": [],
            },
        ),
        Tool(
            name="price_vs_flow_divergence",
            description=(
                "Compare Yahoo Finance price action against UW options flow direction. "
                "When smart money disagrees with price, it may signal a reversal."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "lookback_days": {"type": "integer", "description": "Days of price history (default: 30)", "default": 30},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="institutional_accumulation_detector",
            description=(
                "Detect potential stealth accumulation: combines dark pool buying "
                "at specific price levels with Yahoo price history."
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
            name="analyst_vs_flow",
            description=(
                "Compare Wall Street analyst recommendations (Yahoo) against "
                "actual options flow sentiment (UW). Are traders agreeing or disagreeing?"
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
            name="conviction_matrix",
            description=(
                "Classify the institutional scenario for a ticker by combining dark pool "
                "buy/sell pressure with options flow direction. Distinguishes between: "
                "DIRECTIONAL_LONG (stock + calls), HEDGED_LONG (stock + puts), "
                "COVERED_CALL (stock + selling calls), DIRECTIONAL_SHORT (stock selling + puts), "
                "and MIXED. Reveals the 'why' behind dark pool accumulation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "bull_threshold": {
                        "type": "number",
                        "description": "Dark pool buy ratio above this = bullish (default: 0.6)",
                        "default": 0.6,
                    },
                    "bear_threshold": {
                        "type": "number",
                        "description": "Dark pool buy ratio below this = bearish (default: 0.4)",
                        "default": 0.4,
                    },
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="daily_synthesis",
            description=(
                "Single endpoint for morning briefing. Returns market regime, top bullish and bearish "
                "signal confluence tickers, and watchlist alerts — all in one structured JSON. "
                "Designed for automated workflows; the LLM writes the prose summary, not this tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "watchlist_group": {"type": "string", "description": "Watchlist group to scan (optional, defaults to full watchlist)"},
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    date = arguments.get("date")

    if name == "stock_deep_dive":
        symbol = arguments["symbol"].upper()
        result = {"symbol": symbol}

        # Yahoo Finance data
        try:
            result["yahoo_fundamentals"] = _get_yahoo_data(symbol)
        except Exception as e:
            result["yahoo_fundamentals"] = f"Error: {e}"

        # Yahoo recommendations
        try:
            t = Ticker(symbol)
            recs = t.recommendations
            if recs is not None and not recs.empty:
                result["analyst_recommendations"] = json.loads(
                    recs.tail(5).to_json(orient="records", default_handler=str)
                )
        except Exception:
            pass

        # UW Screener data
        try:
            screener = load_data("screener", date)
            row = screener[screener["ticker"] == symbol]
            if not row.empty:
                cols = ["call_volume", "put_volume", "call_premium", "put_premium",
                        "put_call_ratio", "bullish_premium", "bearish_premium",
                        "iv_rank", "volatility", "iv30d", "implied_move", "implied_move_perc",
                        "total_open_interest", "next_earnings_date"]
                result["uw_screener"] = json.loads(
                    row[[c for c in cols if c in row.columns]].iloc[0].to_json(default_handler=str)
                )
        except Exception:
            pass

        # UW Dark Pool summary
        try:
            dp = load_data("darkpool", date)
            dp_ticker = dp[dp["ticker"] == symbol]
            if not dp_ticker.empty:
                dp_ticker["premium"] = dp_ticker["premium"].astype(float)
                result["uw_dark_pool"] = {
                    "total_premium": float(dp_ticker["premium"].sum()),
                    "total_shares": int(dp_ticker["size"].astype(int).sum()),
                    "trade_count": len(dp_ticker),
                    "avg_price": float(dp_ticker["price"].astype(float).mean()),
                }
        except Exception:
            pass

        # UW OI Changes summary
        try:
            oi = load_data("oi", date)
            oi_ticker = oi[oi["underlying_symbol"] == symbol]
            if not oi_ticker.empty:
                oi_ticker["oi_diff_plain"] = oi_ticker["oi_diff_plain"].astype(float)
                top_oi = oi_ticker.sort_values("oi_diff_plain", ascending=False).head(5)
                result["uw_top_oi_changes"] = json.loads(
                    top_oi[["option_symbol", "strike", "oi_diff_plain", "volume",
                            "avg_price", "dte"]].to_json(orient="records", default_handler=str)
                )
        except Exception:
            pass

        return text_result(result)

    elif name == "earnings_play_analyzer":
        screener = load_data("screener", date)
        screener["next_earnings_date"] = pd.to_datetime(screener["next_earnings_date"], errors="coerce")
        screener["date"] = pd.to_datetime(screener["date"])
        screener["days_to_earnings"] = (screener["next_earnings_date"] - screener["date"]).dt.days

        max_days = arguments.get("days_until_earnings", 14)
        min_iv = arguments.get("min_iv_rank", 40)
        top_n = arguments.get("top_n", 10)

        candidates = screener[
            (screener["days_to_earnings"] > 0) &
            (screener["days_to_earnings"] <= max_days) &
            (screener["iv_rank"].astype(float) >= min_iv)
        ].sort_values("iv_rank", ascending=False).head(top_n)

        results = []
        for _, row in candidates.iterrows():
            ticker = row["ticker"]
            entry = {
                "ticker": ticker,
                "days_to_earnings": int(row["days_to_earnings"]),
                "earnings_date": str(row["next_earnings_date"].date()),
                "iv_rank": float(row["iv_rank"]),
                "implied_move_perc": row.get("implied_move_perc"),
                "put_call_ratio": row.get("put_call_ratio"),
                "bullish_premium": row.get("bullish_premium"),
                "bearish_premium": row.get("bearish_premium"),
            }
            # Enrich with Yahoo data
            try:
                yahoo = _get_yahoo_data(ticker)
                entry["current_price"] = yahoo["current_price"]
                entry["pe_ratio"] = yahoo["pe_ratio"]
                entry["short_percent"] = yahoo["short_percent"]
            except Exception:
                pass

            # Check OI changes for unusual positioning
            try:
                oi = load_data("oi", date)
                oi_t = oi[oi["underlying_symbol"] == ticker]
                if not oi_t.empty:
                    oi_t["oi_diff_plain"] = oi_t["oi_diff_plain"].astype(float)
                    entry["total_oi_increase"] = float(oi_t[oi_t["oi_diff_plain"] > 0]["oi_diff_plain"].sum())
                    entry["total_oi_decrease"] = float(oi_t[oi_t["oi_diff_plain"] < 0]["oi_diff_plain"].sum())
            except Exception:
                pass

            results.append(entry)

        return text_result(results)

    elif name == "price_vs_flow_divergence":
        symbol = arguments["symbol"].upper()
        lookback = arguments.get("lookback_days", 30)

        # Get Yahoo price history
        t = Ticker(symbol)
        hist = t.history(period=f"{lookback}d")
        if hist.empty:
            return text_result(f"No price history found for {symbol}")

        price_change_pct = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100

        # Get UW flow data
        screener = load_data("screener", date)
        row = screener[screener["ticker"] == symbol]

        result = {
            "symbol": symbol,
            "price_start": float(hist["Close"].iloc[0]),
            "price_end": float(hist["Close"].iloc[-1]),
            "price_change_pct": round(price_change_pct, 2),
            "period_high": float(hist["High"].max()),
            "period_low": float(hist["Low"].min()),
        }

        if not row.empty:
            r = row.iloc[0]
            bull = float(r.get("bullish_premium", 0) or 0)
            bear = float(r.get("bearish_premium", 0) or 0)
            net = bull - bear
            result["bullish_premium"] = bull
            result["bearish_premium"] = bear
            result["net_premium_flow"] = net
            result["flow_direction"] = "bullish" if net > 0 else "bearish"
            result["put_call_ratio"] = r.get("put_call_ratio")
            result["iv_rank"] = r.get("iv_rank")

            # Detect divergence
            price_bullish = price_change_pct > 0
            flow_bullish = net > 0
            if price_bullish != flow_bullish:
                result["divergence"] = True
                result["divergence_signal"] = (
                    f"DIVERGENCE: Price is {'up' if price_bullish else 'down'} "
                    f"{abs(price_change_pct):.1f}% but options flow is "
                    f"{'bullish' if flow_bullish else 'bearish'} "
                    f"(net flow: ${net:,.0f})"
                )
            else:
                result["divergence"] = False
                result["divergence_signal"] = "Price and flow are aligned"

        return text_result(result)

    elif name == "institutional_accumulation_detector":
        symbol = arguments["symbol"].upper()

        # Dark pool data
        try:
            dp = load_data("darkpool", date)
            dp_ticker = dp[dp["ticker"] == symbol]
        except Exception:
            return text_result(f"No dark pool data for {symbol}")

        if dp_ticker.empty:
            return text_result(f"No dark pool trades found for {symbol}")

        dp_ticker = dp_ticker.copy()
        dp_ticker["price"] = dp_ticker["price"].astype(float)
        dp_ticker["premium"] = dp_ticker["premium"].astype(float)
        dp_ticker["size"] = dp_ticker["size"].astype(int)
        dp_ticker["nbbo_bid"] = dp_ticker["nbbo_bid"].astype(float)
        dp_ticker["nbbo_ask"] = dp_ticker["nbbo_ask"].astype(float)
        dp_ticker["nbbo_mid"] = (dp_ticker["nbbo_bid"] + dp_ticker["nbbo_ask"]) / 2

        # Classify trades: above mid = buying pressure, below mid = selling pressure
        dp_ticker["side"] = "neutral"
        dp_ticker.loc[dp_ticker["price"] >= dp_ticker["nbbo_mid"], "side"] = "buy"
        dp_ticker.loc[dp_ticker["price"] < dp_ticker["nbbo_mid"], "side"] = "sell"

        buy_volume = int(dp_ticker[dp_ticker["side"] == "buy"]["size"].sum())
        sell_volume = int(dp_ticker[dp_ticker["side"] == "sell"]["size"].sum())
        total_volume = buy_volume + sell_volume

        # Yahoo price context
        t = Ticker(symbol)
        hist = t.history(period="30d")
        price_30d_change = None
        if not hist.empty and len(hist) > 1:
            price_30d_change = round(
                ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100, 2
            )

        result = {
            "symbol": symbol,
            "dark_pool_trades": len(dp_ticker),
            "total_dp_volume": total_volume,
            "total_dp_premium": float(dp_ticker["premium"].sum()),
            "buy_side_volume": buy_volume,
            "sell_side_volume": sell_volume,
            "buy_sell_ratio": round(buy_volume / max(sell_volume, 1), 2),
            "avg_trade_price": round(float(dp_ticker["price"].mean()), 2),
            "vwap": round(float((dp_ticker["price"] * dp_ticker["size"]).sum() / dp_ticker["size"].sum()), 2),
            "price_30d_change_pct": price_30d_change,
        }

        if buy_volume > sell_volume * 1.5:
            result["signal"] = "ACCUMULATION \u2014 dark pool buy volume significantly exceeds sell volume"
        elif sell_volume > buy_volume * 1.5:
            result["signal"] = "DISTRIBUTION \u2014 dark pool sell volume significantly exceeds buy volume"
        else:
            result["signal"] = "NEUTRAL \u2014 balanced dark pool activity"

        # Top price levels
        levels = dp_ticker.groupby(dp_ticker["price"].round(2)).agg(
            shares=("size", "sum"), trades=("size", "count"), premium=("premium", "sum")
        ).sort_values("shares", ascending=False).head(5)
        result["top_price_levels"] = json.loads(levels.reset_index().to_json(orient="records"))

        return text_result(result)

    elif name == "analyst_vs_flow":
        symbol = arguments["symbol"].upper()

        # Yahoo analyst recs
        t = Ticker(symbol)
        recs = t.recommendations
        analyst_data = {}
        if recs is not None and not recs.empty:
            latest = recs.iloc[-1]
            analyst_data = {
                "period": str(latest.name) if hasattr(latest, "name") else "latest",
                "strongBuy": int(latest.get("strongBuy", 0)),
                "buy": int(latest.get("buy", 0)),
                "hold": int(latest.get("hold", 0)),
                "sell": int(latest.get("sell", 0)),
                "strongSell": int(latest.get("strongSell", 0)),
            }
            total_recs = sum(analyst_data[k] for k in ["strongBuy", "buy", "hold", "sell", "strongSell"])
            if total_recs > 0:
                bull_pct = (analyst_data["strongBuy"] + analyst_data["buy"]) / total_recs * 100
                analyst_data["bullish_pct"] = round(bull_pct, 1)
                analyst_data["analyst_sentiment"] = "bullish" if bull_pct > 60 else ("bearish" if bull_pct < 40 else "mixed")

        # UW flow data
        screener = load_data("screener", date)
        row = screener[screener["ticker"] == symbol]
        flow_data = {}
        if not row.empty:
            r = row.iloc[0]
            bull = float(r.get("bullish_premium", 0) or 0)
            bear = float(r.get("bearish_premium", 0) or 0)
            flow_data = {
                "bullish_premium": bull,
                "bearish_premium": bear,
                "net_flow": bull - bear,
                "put_call_ratio": r.get("put_call_ratio"),
                "flow_sentiment": "bullish" if bull > bear else "bearish",
            }

        result = {
            "symbol": symbol,
            "analyst_recommendations": analyst_data,
            "options_flow": flow_data,
        }

        # Compare
        if analyst_data.get("analyst_sentiment") and flow_data.get("flow_sentiment"):
            agree = analyst_data["analyst_sentiment"] == flow_data["flow_sentiment"]
            result["agreement"] = agree
            if not agree:
                result["signal"] = (
                    f"DISAGREEMENT: Analysts are {analyst_data['analyst_sentiment']} "
                    f"({analyst_data.get('bullish_pct', '?')}% buy) but options flow is "
                    f"{flow_data['flow_sentiment']} (net: ${flow_data['net_flow']:,.0f})"
                )
            else:
                result["signal"] = (
                    f"ALIGNED: Both analysts and options flow are {analyst_data['analyst_sentiment']}"
                )

        return text_result(result)

    elif name == "conviction_matrix":
        symbol = arguments["symbol"].upper().strip()
        bull_threshold = arguments.get("bull_threshold", 0.6)
        bear_threshold = arguments.get("bear_threshold", 0.4)

        dp_stats = None
        dp_buy_ratio = None
        try:
            dp = load_data("darkpool", date)
            dp_sym = dp[dp["ticker"] == symbol].copy()
            if not dp_sym.empty:
                dp_sym["price"] = dp_sym["price"].astype(float)
                dp_sym["nbbo_bid"] = dp_sym["nbbo_bid"].astype(float)
                dp_sym["nbbo_ask"] = dp_sym["nbbo_ask"].astype(float)
                dp_sym["size"] = dp_sym["size"].astype(int)
                dp_sym["nbbo_mid"] = (dp_sym["nbbo_bid"] + dp_sym["nbbo_ask"]) / 2
                dp_sym["dp_side"] = "sell"
                dp_sym.loc[dp_sym["price"] >= dp_sym["nbbo_mid"], "dp_side"] = "buy"
                buy_vol = int(dp_sym[dp_sym["dp_side"] == "buy"]["size"].sum())
                sell_vol = int(dp_sym[dp_sym["dp_side"] == "sell"]["size"].sum())
                total_dp_vol = buy_vol + sell_vol
                dp_buy_ratio = buy_vol / total_dp_vol if total_dp_vol > 0 else 0.5
                dp_stats = {
                    "trades": len(dp_sym),
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                    "buy_ratio": round(dp_buy_ratio, 3),
                }
        except Exception:
            pass

        call_ask_vol = call_bid_vol = put_ask_vol = put_bid_vol = 0.0
        opts_stats = None
        try:
            opts_cols = ["underlying_symbol", "option_type", "side", "size"]
            opts = load_data("options", date, usecols=opts_cols)
            opts_sym = opts[opts["underlying_symbol"] == symbol].copy()
            if not opts_sym.empty:
                opts_sym["size"] = opts_sym["size"].astype(float)
                calls = opts_sym[opts_sym["option_type"] == "call"]
                puts = opts_sym[opts_sym["option_type"] == "put"]
                call_ask_vol = float(calls[calls["side"] == "ask"]["size"].sum())
                call_bid_vol = float(calls[calls["side"] == "bid"]["size"].sum())
                put_ask_vol = float(puts[puts["side"] == "ask"]["size"].sum())
                put_bid_vol = float(puts[puts["side"] == "bid"]["size"].sum())
                opts_stats = {
                    "call_ask_volume": int(call_ask_vol),
                    "call_bid_volume": int(call_bid_vol),
                    "put_ask_volume": int(put_ask_vol),
                    "put_bid_volume": int(put_bid_vol),
                }
        except Exception:
            pass

        if dp_buy_ratio is None and opts_stats is None:
            scenario = "NO_DATA"
            explanation = "No dark pool or options data found for this symbol."
            confidence = 0.0
        elif dp_buy_ratio is None:
            scenario = "PARTIAL_DATA"
            explanation = "Options data available but no dark pool activity found."
            confidence = 0.0
        else:
            dp_bullish = dp_buy_ratio > bull_threshold
            dp_bearish = dp_buy_ratio < bear_threshold

            if dp_bullish:
                if call_ask_vol > call_bid_vol and call_ask_vol > put_ask_vol:
                    scenario = "DIRECTIONAL_LONG"
                    explanation = "Dark pool buying + aggressive call purchases — institutional directional bet."
                elif put_ask_vol > put_bid_vol and put_ask_vol >= call_ask_vol:
                    scenario = "HEDGED_LONG"
                    explanation = "Dark pool buying + put protection — institution buying stock and hedging downside."
                elif call_bid_vol > call_ask_vol:
                    scenario = "COVERED_CALL"
                    explanation = "Dark pool buying + call selling — yield enhancement strategy, capping upside."
                else:
                    scenario = "BULLISH_ACCUMULATION"
                    explanation = "Dark pool buying with mixed options activity — directional bias unclear."
            elif dp_bearish:
                if put_ask_vol > put_bid_vol:
                    scenario = "DIRECTIONAL_SHORT"
                    explanation = "Dark pool selling + put buying — institutional directional bear bet."
                else:
                    scenario = "DISTRIBUTION"
                    explanation = "Dark pool selling with mixed options activity."
            else:
                scenario = "MIXED"
                explanation = "Balanced dark pool activity — no clear institutional directional bias."

            dp_confidence = abs(dp_buy_ratio - 0.5) * 2
            opts_confidence = 0.0
            if scenario in ("DIRECTIONAL_LONG", "COVERED_CALL"):
                total_call = call_ask_vol + call_bid_vol
                if total_call > 0:
                    opts_confidence = abs(call_ask_vol - call_bid_vol) / total_call
            elif scenario in ("HEDGED_LONG", "DIRECTIONAL_SHORT"):
                total_put = put_ask_vol + put_bid_vol
                if total_put > 0:
                    opts_confidence = abs(put_ask_vol - put_bid_vol) / total_put
            confidence = round((dp_confidence + opts_confidence) / 2 * 100, 1)

        return text_result({
            "symbol": symbol,
            "scenario": scenario,
            "confidence_pct": confidence,
            "explanation": explanation,
            "dark_pool": dp_stats,
            "options_flow": opts_stats,
            "thresholds": {"bull": bull_threshold, "bear": bear_threshold},
        })

    elif name == "daily_synthesis":
        from servers.risk import compute_market_regime, compute_signal_confluence
        from servers.watchlist import compute_watchlist_alerts

        watchlist_group = arguments.get("watchlist_group")

        regime = None
        bullish_signals = None
        bearish_signals = None
        watchlist_data = None

        try:
            regime = compute_market_regime(date)
        except Exception as e:
            regime = {"error": str(e)}

        try:
            bullish_signals = compute_signal_confluence(date, direction="bullish", top_n=10)
        except Exception as e:
            bullish_signals = {"error": str(e)}

        try:
            bearish_signals = compute_signal_confluence(date, direction="bearish", top_n=10)
        except Exception as e:
            bearish_signals = {"error": str(e)}

        try:
            watchlist_data = compute_watchlist_alerts(date, group=watchlist_group)
        except Exception as e:
            watchlist_data = {"error": str(e)}

        return text_result({
            "date": date or "latest",
            "market_regime": regime,
            "top_bullish_signals": bullish_signals,
            "top_bearish_signals": bearish_signals,
            "watchlist_alerts": watchlist_data,
        })

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
