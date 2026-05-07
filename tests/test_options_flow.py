"""Tests for the new servers.options_flow (Phase 2 — 9 trade-level tools).

The structural tools (iv_term_structure, term_skew, front_end_iv_ratio,
today_gamma_flip, gamma_exposure_profile) moved to servers.options_structure.
This server now houses only trade-level flow tools.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures import synthesise

pytestmark = pytest.mark.unit


def _opra(root: str, expiry: date, cp: str, strike: float) -> str:
    return f"{root}{expiry.strftime('%y%m%d')}{cp}{int(strike * 1000):08d}"


@pytest.fixture
def options_parquet_basic(stocks_dir: Path) -> Path:
    """Minimal options Parquet so each trade-level tool can dispatch."""
    expiry = date.today() + timedelta(days=14)
    rows = []
    for ticker, sector in [("SPY", "Index"), ("AAPL", "Technology"), ("XOM", "Energy")]:
        for cp, opt_type, side in [("C", "call", "ask"), ("P", "put", "bid")]:
            for strike in (95.0, 100.0, 105.0):
                rows.append({
                    "option_symbol": _opra(ticker, expiry, cp, strike),
                    "underlying_symbol": ticker,
                    "executed_at": "2026-03-17 10:00:00",
                    "strike": strike,
                    "option_type": opt_type,
                    "expiry": expiry.strftime("%Y-%m-%d"),
                    "underlying_price": 100.0,
                    "price": 5.0,
                    "size": 100,
                    "premium": 50000.0,
                    "volume": 100,
                    "open_interest": 1000,
                    "implied_volatility": 0.25,
                    "delta": 0.5 if cp == "C" else -0.5,
                    "theta": -0.05,
                    "gamma": 0.01,
                    "vega": 0.10,
                    "side": side,
                    "sector": sector,
                })
    df = pd.DataFrame(rows)
    return synthesise.write_options_parquet(stocks_dir, "2026-03-17", df)


@pytest.mark.asyncio
async def test_server_lists_nine_tools():
    """Acceptance: uw-options-flow exposes exactly the 9 trade-level tools."""
    from servers.options_flow import list_tools

    tools = await list_tools()
    names = {t.name for t in tools}
    expected = {
        "top_premium_trades",
        "unusual_volume_scanner",
        "sweep_detector",
        "iv_outliers",
        "greek_screener",
        "expiry_heatmap",
        "sector_flow_summary",
        "sector_flow_persistence",
        "dte_volume_share",
    }
    assert names == expected


@pytest.mark.asyncio
async def test_structural_tools_no_longer_dispatch():
    """Tools that moved to options_structure must NOT respond from this server."""
    from servers.options_flow import call_tool

    for moved in ("iv_term_structure", "term_skew", "front_end_iv_ratio",
                  "today_gamma_flip", "gamma_exposure_profile"):
        result = await call_tool(moved, {"symbol": "SPY"})
        text = result[0].text
        assert "Unknown tool" in text or "error" in text.lower(), (
            f"{moved} should not be dispatched from options_flow"
        )


@pytest.mark.asyncio
class TestTradeLevelToolsDispatch:
    """Smoke tests confirming each of the 9 trade-level tools dispatches."""

    async def test_top_premium_trades(self, options_parquet_basic: Path):
        from servers.options_flow import call_tool

        result = await call_tool("top_premium_trades", {"top_n": 3})
        text = result[0].text
        assert "premium" in text.lower()

    async def test_unusual_volume_scanner(self, options_parquet_basic: Path):
        from servers.options_flow import call_tool

        result = await call_tool("unusual_volume_scanner", {"top_n": 3, "min_volume": 1, "min_vol_oi_ratio": 0.01})
        # Just verify no crash + valid response
        assert len(result) == 1

    async def test_sweep_detector(self, options_parquet_basic: Path):
        from servers.options_flow import call_tool

        result = await call_tool("sweep_detector", {"top_n": 3, "min_premium": 1000})
        assert len(result) == 1

    async def test_iv_outliers(self, options_parquet_basic: Path):
        from servers.options_flow import call_tool

        result = await call_tool("iv_outliers", {"top_n": 3, "min_iv": 0.1, "min_volume": 1})
        assert len(result) == 1

    async def test_greek_screener(self, options_parquet_basic: Path):
        from servers.options_flow import call_tool

        result = await call_tool("greek_screener", {"top_n": 3})
        assert len(result) == 1

    async def test_expiry_heatmap(self, options_parquet_basic: Path):
        from servers.options_flow import call_tool

        result = await call_tool("expiry_heatmap", {"top_n": 3})
        assert len(result) == 1

    async def test_sector_flow_summary(self, options_parquet_basic: Path):
        from servers.options_flow import call_tool

        result = await call_tool("sector_flow_summary", {})
        text = result[0].text
        # Should mention at least one sector from fixture
        assert any(s in text for s in ("Index", "Technology", "Energy"))

    async def test_dte_volume_share(self, options_parquet_basic: Path):
        from servers.options_flow import call_tool

        result = await call_tool("dte_volume_share", {})
        payload = json.loads(result[0].text)
        # Should have some DTE bucket populated
        for key in ("share_0dte", "share_weeklies", "share_monthlies", "share_leaps"):
            assert key in payload
