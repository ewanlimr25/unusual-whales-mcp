"""Tests for the new pin_risk_screener tool — Phase 4 audit §3.8.

Ranks tickers within N days of monthly OPEX by gamma-weighted distance × OI
mass. Uses Ni-Pearson-Poteshman 2005 / Avellaneda-Lipkin 2003 framework.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures import synthesise

pytestmark = pytest.mark.unit


def _opra(root: str, yymmdd: str, cp: str, strike: float) -> str:
    return f"{root}{yymmdd}{cp}{int(strike * 1000):08d}"


@pytest.fixture
def pin_oi_parquet(stocks_dir: Path) -> Path:
    """Synthesise OI rows for three pinning archetypes:

    AAPL — high OI cluster at $150, spot $149.50, DTE=2 → STRONG pin candidate
    MSFT — modest OI on a strike $5 from spot, DTE=2 → weak pin candidate
    NVDA — high OI but DTE=30 (no near OPEX) → must be filtered out
    """
    yymmdd_near = "260320"  # near OPEX
    yymmdd_far = "260417"   # ~30 DTE away
    rows: list[dict] = []

    # AAPL — concentrated near-spot calls + puts
    for strike, oi in [(145.0, 500), (150.0, 5000), (155.0, 500)]:
        for cp, opt_type in [("C", "call"), ("P", "put")]:
            rows.append({
                "option_symbol": _opra("AAPL", yymmdd_near, cp, strike),
                "underlying_symbol": "AAPL",
                "strike": strike,
                "last_oi": int(oi * 0.8),
                "curr_oi": oi,
                "oi_diff_plain": float(oi * 0.2),
                "volume": 100,
                "oi_change": 0.25,
                "avg_price": 1.0,
                "prev_total_premium": 50000.0,
                "prev_ask_volume": 50.0,
                "prev_bid_volume": 50.0,
                "sector": "Technology",
                "stock_price": 149.50,
                "dte": 2,
            })

    # MSFT — moderate OI, strike $5 from spot
    rows.append({
        "option_symbol": _opra("MSFT", yymmdd_near, "C", 250.0),
        "underlying_symbol": "MSFT",
        "strike": 250.0,
        "last_oi": 800,
        "curr_oi": 1000,
        "oi_diff_plain": 200.0,
        "volume": 100,
        "oi_change": 0.25,
        "avg_price": 1.0,
        "prev_total_premium": 50000.0,
        "prev_ask_volume": 50.0,
        "prev_bid_volume": 50.0,
        "sector": "Technology",
        "stock_price": 245.0,  # 2% from strike
        "dte": 2,
    })

    # NVDA — far DTE → must be excluded by default `dte_max=7`
    rows.append({
        "option_symbol": _opra("NVDA", yymmdd_far, "C", 800.0),
        "underlying_symbol": "NVDA",
        "strike": 800.0,
        "last_oi": 5000,
        "curr_oi": 10000,
        "oi_diff_plain": 5000.0,
        "volume": 100,
        "oi_change": 1.0,
        "avg_price": 1.0,
        "prev_total_premium": 50000.0,
        "prev_ask_volume": 50.0,
        "prev_bid_volume": 50.0,
        "sector": "Technology",
        "stock_price": 800.0,
        "dte": 30,
    })

    df = pd.DataFrame(rows)
    return synthesise.write_oi_parquet(stocks_dir, "2026-03-18", df)


@pytest.mark.asyncio
class TestPinRiskScreener:
    async def test_returns_ranked_tickers(self, pin_oi_parquet: Path):
        from servers.oi_changes import call_tool

        result = await call_tool("pin_risk_screener", {})
        payload = json.loads(result[0].text)
        assert "results" in payload
        results = payload["results"]
        assert len(results) >= 1

        first = results[0]
        for field in ("ticker", "nearest_high_oi_strike", "pin_distance_pct",
                      "dte_to_opex", "pin_score", "spot"):
            assert field in first, f"Missing field {field}"

    async def test_aapl_outranks_msft(self, pin_oi_parquet: Path):
        """AAPL has 5000 OI cluster within 0.3% of spot vs MSFT's 1000 OI at 2%.
        Both have DTE=2. AAPL must rank higher."""
        from servers.oi_changes import call_tool

        result = await call_tool("pin_risk_screener", {})
        payload = json.loads(result[0].text)
        results = payload["results"]
        tickers_in_order = [r["ticker"] for r in results]
        assert tickers_in_order.index("AAPL") < tickers_in_order.index("MSFT")

    async def test_far_dte_excluded(self, pin_oi_parquet: Path):
        """NVDA has DTE=30 — must be filtered out by default dte_max=7."""
        from servers.oi_changes import call_tool

        result = await call_tool("pin_risk_screener", {})
        payload = json.loads(result[0].text)
        tickers = [r["ticker"] for r in payload["results"]]
        assert "NVDA" not in tickers

    async def test_dte_max_override_includes_far(self, pin_oi_parquet: Path):
        from servers.oi_changes import call_tool

        result = await call_tool("pin_risk_screener", {"dte_max": 60})
        payload = json.loads(result[0].text)
        tickers = [r["ticker"] for r in payload["results"]]
        assert "NVDA" in tickers

    async def test_max_distance_pct_filter(self, pin_oi_parquet: Path):
        """MSFT's strike is 2% from spot. With max_distance_pct=1, MSFT excluded."""
        from servers.oi_changes import call_tool

        result = await call_tool("pin_risk_screener", {"max_distance_pct": 1.0})
        payload = json.loads(result[0].text)
        tickers = [r["ticker"] for r in payload["results"]]
        assert "MSFT" not in tickers
        assert "AAPL" in tickers

    async def test_top_n_limits(self, pin_oi_parquet: Path):
        from servers.oi_changes import call_tool

        result = await call_tool("pin_risk_screener", {"top_n": 1})
        payload = json.loads(result[0].text)
        assert len(payload["results"]) <= 1

    async def test_invalid_top_n_raises(self):
        from servers.oi_changes import call_tool

        with pytest.raises(ValueError, match="top_n"):
            await call_tool("pin_risk_screener", {"top_n": -1})

    async def test_dte_max_zero_is_valid(self, pin_oi_parquet):
        """OPEX Friday afternoon use case: query expiring-today contracts only."""
        from servers.oi_changes import call_tool

        # Should not raise; tool handles dte_max=0 even if no DTE=0 in fixture
        result = await call_tool("pin_risk_screener", {"dte_max": 0})
        # Just confirm it returns a structured response
        json.loads(result[0].text)

    async def test_invalid_dte_max_negative_raises(self):
        from servers.oi_changes import call_tool

        with pytest.raises(ValueError, match="dte_max"):
            await call_tool("pin_risk_screener", {"dte_max": -1})

    async def test_invalid_dte_max_over_cap_raises(self):
        from servers.oi_changes import call_tool

        with pytest.raises(ValueError, match="dte_max"):
            await call_tool("pin_risk_screener", {"dte_max": 366})

    async def test_aggregated_no_raw_trades(self, pin_oi_parquet: Path):
        """Strong key-set assertion — guards against future regressions adding
        per-contract fields (option_symbol, oi_diff_plain, etc.) to output."""
        from servers.oi_changes import call_tool

        result = await call_tool("pin_risk_screener", {})
        payload = json.loads(result[0].text)
        allowed_keys = {
            "ticker", "spot", "nearest_high_oi_strike", "pin_distance_pct",
            "top_strike_oi", "total_oi_in_window", "dte_to_opex", "pin_score",
        }
        for row in payload["results"]:
            assert set(row.keys()) <= allowed_keys, (
                f"Unexpected per-contract keys leaked: {set(row.keys()) - allowed_keys}"
            )
        # Belt-and-braces: no raw timestamp or option_symbol leakage
        text = result[0].text
        assert "executed_at" not in text
        assert "option_symbol" not in text

    async def test_caveat_field_present(self, pin_oi_parquet: Path):
        from servers.oi_changes import call_tool

        result = await call_tool("pin_risk_screener", {})
        payload = json.loads(result[0].text)
        assert "caveat" in payload or "note" in payload
