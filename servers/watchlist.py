"""MCP Server: Watchlist management and filtered scanning."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from shared.data_loader import load_data
from shared.server_utils import text_result, df_to_result

server = Server("uw-watchlist")

WATCHLIST_FILE = Path(__file__).parent.parent / "watchlist.json"


def _load_watchlist() -> dict:
    """Load watchlist from file."""
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text())
    return {"tickers": [], "groups": {}}


def _save_watchlist(data: dict):
    """Save watchlist to file."""
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2))


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="manage_watchlist",
            description=(
                "Add, remove, or list tickers in your watchlist. "
                "Supports groups (e.g., 'earnings_plays', 'momentum', 'core_holdings')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "list", "clear"],
                        "description": "Action to perform",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ticker symbols to add/remove (optional for list/clear)",
                    },
                    "group": {
                        "type": "string",
                        "description": "Group name (optional). Tickers are added to both the main list and the group.",
                    },
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="watchlist_scan",
            description=(
                "Run a comprehensive scan on all watchlist tickers. "
                "Shows options flow, dark pool activity, OI changes, IV rank, "
                "and volume anomalies — but only for tickers you care about."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "group": {"type": "string", "description": "Scan only a specific group (optional)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="watchlist_alerts",
            description=(
                "Flag only unusual/notable activity on watchlist tickers. "
                "Highlights volume spikes, IV rank extremes, large dark pool trades, "
                "and significant OI changes. Cuts through noise to show what matters."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD). Defaults to latest."},
                    "group": {"type": "string", "description": "Check only a specific group (optional)"},
                    "min_volume_ratio": {"type": "number", "description": "Min volume/avg ratio for alert (default: 1.5)", "default": 1.5},
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    if name == "manage_watchlist":
        wl = _load_watchlist()
        action = arguments["action"]
        tickers = [t.upper() for t in arguments.get("tickers", [])]
        group = arguments.get("group")

        if action == "add":
            if not tickers:
                return text_result("No tickers specified to add")
            for t in tickers:
                if t not in wl["tickers"]:
                    wl["tickers"].append(t)
            if group:
                if group not in wl["groups"]:
                    wl["groups"][group] = []
                for t in tickers:
                    if t not in wl["groups"][group]:
                        wl["groups"][group].append(t)
            _save_watchlist(wl)
            return text_result(f"Added {tickers}. Watchlist: {len(wl['tickers'])} tickers, {len(wl['groups'])} groups.")

        elif action == "remove":
            for t in tickers:
                if t in wl["tickers"]:
                    wl["tickers"].remove(t)
                for g in wl["groups"].values():
                    if t in g:
                        g.remove(t)
            _save_watchlist(wl)
            return text_result(f"Removed {tickers}. Watchlist: {len(wl['tickers'])} tickers.")

        elif action == "list":
            return text_result(wl)

        elif action == "clear":
            if group:
                wl["groups"].pop(group, None)
                _save_watchlist(wl)
                return text_result(f"Cleared group '{group}'")
            wl = {"tickers": [], "groups": {}}
            _save_watchlist(wl)
            return text_result("Watchlist cleared")

    elif name == "watchlist_scan":
        wl = _load_watchlist()
        group = arguments.get("group")
        tickers = wl["groups"].get(group, []) if group else wl["tickers"]
        date = arguments.get("date")

        if not tickers:
            return text_result("Watchlist is empty. Use manage_watchlist to add tickers first.")

        results = []
        # Load all data sources once
        try:
            screener = load_data("screener", date)
        except Exception:
            screener = None
        try:
            dp = load_data("darkpool", date)
            dp["premium"] = dp["premium"].astype(float)
        except Exception:
            dp = None
        try:
            oi = load_data("oi", date)
            oi["oi_diff_plain"] = oi["oi_diff_plain"].astype(float)
        except Exception:
            oi = None

        for ticker in tickers:
            entry = {"ticker": ticker}

            # Screener data
            if screener is not None:
                row = screener[screener["ticker"] == ticker]
                if not row.empty:
                    r = row.iloc[0]
                    entry["close"] = r.get("close")
                    entry["call_volume"] = r.get("call_volume")
                    entry["put_volume"] = r.get("put_volume")
                    entry["put_call_ratio"] = r.get("put_call_ratio")
                    entry["iv_rank"] = r.get("iv_rank")
                    bull = float(r.get("bullish_premium", 0) or 0)
                    bear = float(r.get("bearish_premium", 0) or 0)
                    entry["net_flow"] = bull - bear
                    entry["flow_direction"] = "bullish" if bull > bear else "bearish"
                    # Volume vs average
                    total_vol = float(r.get("call_volume", 0) or 0) + float(r.get("put_volume", 0) or 0)
                    avg_vol = float(r.get("avg_30_day_call_volume", 0) or 0) + float(r.get("avg_30_day_put_volume", 0) or 0)
                    entry["volume_ratio"] = round(total_vol / avg_vol, 2) if avg_vol > 0 else None

            # Dark pool summary
            if dp is not None:
                dp_t = dp[dp["ticker"] == ticker]
                if not dp_t.empty:
                    entry["dp_trades"] = len(dp_t)
                    entry["dp_total_premium"] = float(dp_t["premium"].sum())
                    entry["dp_total_shares"] = int(dp_t["size"].astype(int).sum())

            # OI changes summary
            if oi is not None:
                oi_t = oi[oi["underlying_symbol"] == ticker]
                if not oi_t.empty:
                    entry["oi_net_change"] = float(oi_t["oi_diff_plain"].sum())
                    entry["oi_contracts_increasing"] = int((oi_t["oi_diff_plain"] > 0).sum())

            results.append(entry)

        return text_result(results)

    elif name == "watchlist_alerts":
        wl = _load_watchlist()
        group = arguments.get("group")
        tickers = wl["groups"].get(group, []) if group else wl["tickers"]
        date = arguments.get("date")
        min_vol_ratio = arguments.get("min_volume_ratio", 1.5)

        if not tickers:
            return text_result("Watchlist is empty. Use manage_watchlist to add tickers first.")

        alerts = []

        try:
            screener = load_data("screener", date)
        except Exception:
            screener = None
        try:
            dp = load_data("darkpool", date)
            dp["premium"] = dp["premium"].astype(float)
        except Exception:
            dp = None
        try:
            oi = load_data("oi", date)
            oi["oi_diff_plain"] = oi["oi_diff_plain"].astype(float)
        except Exception:
            oi = None

        for ticker in tickers:
            ticker_alerts = []

            if screener is not None:
                row = screener[screener["ticker"] == ticker]
                if not row.empty:
                    r = row.iloc[0]
                    # Volume spike
                    total_vol = float(r.get("call_volume", 0) or 0) + float(r.get("put_volume", 0) or 0)
                    avg_vol = float(r.get("avg_30_day_call_volume", 0) or 0) + float(r.get("avg_30_day_put_volume", 0) or 0)
                    if avg_vol > 0:
                        vol_ratio = total_vol / avg_vol
                        if vol_ratio >= min_vol_ratio:
                            ticker_alerts.append({
                                "type": "VOLUME_SPIKE",
                                "detail": f"{vol_ratio:.1f}x average options volume",
                                "severity": "high" if vol_ratio >= 3 else "medium",
                            })
                    # IV rank extreme
                    iv_rank = float(r.get("iv_rank", 0) or 0)
                    if iv_rank >= 80:
                        ticker_alerts.append({
                            "type": "HIGH_IV_RANK",
                            "detail": f"IV rank at {iv_rank:.1f} — options are expensive",
                            "severity": "medium",
                        })
                    elif iv_rank <= 10 and iv_rank > 0:
                        ticker_alerts.append({
                            "type": "LOW_IV_RANK",
                            "detail": f"IV rank at {iv_rank:.1f} — options are cheap",
                            "severity": "medium",
                        })
                    # Extreme P/C ratio
                    pcr = float(r.get("put_call_ratio", 0) or 0)
                    if pcr >= 3:
                        ticker_alerts.append({
                            "type": "HIGH_PUT_CALL",
                            "detail": f"P/C ratio {pcr:.2f} — heavy put activity",
                            "severity": "high",
                        })
                    elif pcr <= 0.3 and pcr > 0:
                        ticker_alerts.append({
                            "type": "LOW_PUT_CALL",
                            "detail": f"P/C ratio {pcr:.2f} — heavy call activity",
                            "severity": "medium",
                        })

            # Large dark pool activity
            if dp is not None:
                dp_t = dp[dp["ticker"] == ticker]
                if not dp_t.empty:
                    max_trade = dp_t["premium"].max()
                    if max_trade >= 1_000_000:
                        ticker_alerts.append({
                            "type": "LARGE_DARK_POOL",
                            "detail": f"${max_trade:,.0f} single dark pool trade, {len(dp_t)} total trades",
                            "severity": "high",
                        })

            # Significant OI change
            if oi is not None:
                oi_t = oi[oi["underlying_symbol"] == ticker]
                if not oi_t.empty:
                    net = float(oi_t["oi_diff_plain"].sum())
                    if abs(net) >= 10000:
                        direction = "increasing" if net > 0 else "decreasing"
                        ticker_alerts.append({
                            "type": "OI_SHIFT",
                            "detail": f"Net OI {direction} by {abs(net):,.0f} contracts",
                            "severity": "high" if abs(net) >= 50000 else "medium",
                        })

            if ticker_alerts:
                alerts.append({
                    "ticker": ticker,
                    "alert_count": len(ticker_alerts),
                    "alerts": ticker_alerts,
                })

        if not alerts:
            return text_result("No unusual activity detected on watchlist tickers today.")

        # Sort by number of alerts (most interesting first)
        alerts.sort(key=lambda x: x["alert_count"], reverse=True)
        return text_result(alerts)

    return text_result(f"Unknown tool: {name}")


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
