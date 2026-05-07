"""Tests for servers.oi_changes.

Covers:
  - The OPRA `str.contains("C")` regression in `smart_positioning` (oi_changes.py:165)
    and `position_rolling_detector` (oi_changes.py:196). Synthetic data includes
    MCD puts which the legacy code misclassified as calls.
  - The new `opex_concentration` tool (Phase 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# OPRA regression tests for the str.contains("C") bug
# ============================================================================


@pytest.mark.regression
class TestSmartPositioningOpraBug:
    """The legacy `df["option_symbol"].str.contains("C")` at oi_changes.py:165
    misclassified MCD puts as calls because 'C' is in 'MCD'.
    After the fix, classify_option_types() reads the OPRA C/P at its proper position.
    """

    @pytest.mark.asyncio
    async def test_mcd_puts_not_classified_as_calls(self, oi_parquet: Path):
        from servers.oi_changes import call_tool

        result = await call_tool(
            "smart_positioning",
            {"symbol": "MCD", "min_oi_change": 100},
        )
        payload = json.loads(result[0].text)
        records = payload if isinstance(payload, list) else payload.get("records", [])

        # Every MCD row in the fixture is a put. Direction inference uses
        # option_type × ask_vs_bid. The ask-heavy MCD puts in the fixture are
        # bearish puts (ask-side put buying = bearish). If misclassified as calls,
        # the same ask-heavy reading would produce "bullish" — the inverse.
        mcd_rows = [r for r in records if r.get("underlying_symbol") == "MCD"]
        assert len(mcd_rows) > 0, "No MCD rows returned"
        for row in mcd_rows:
            assert row["inferred_direction"] == "bearish", (
                f"MCD put with ask-heavy volume should be bearish, "
                f"got {row['inferred_direction']} — OPRA bug is back"
            )


@pytest.mark.regression
class TestPositionRollingDetectorOpraBug:
    @pytest.mark.asyncio
    async def test_googl_call_roll_detected_not_mcd_phantom(self, oi_parquet: Path):
        """The synthetic data has a real call roll on GOOGL (near calls down,
        far calls up). MCD only has puts and they should NOT appear as calls.
        """
        from servers.oi_changes import call_tool

        result = await call_tool(
            "position_rolling_detector",
            {"threshold": 500},
        )
        payload = json.loads(result[0].text)
        rolls = payload.get("results", [])

        # GOOGL call roll must be detected
        googl_call_rolls = [
            r for r in rolls
            if r["symbol"] == "GOOGL" and r["option_type"] == "call"
        ]
        assert len(googl_call_rolls) == 1, "Expected one GOOGL call roll"

        # MCD has only puts in the fixture — must not appear as a call roll
        mcd_call_rolls = [
            r for r in rolls
            if r["symbol"] == "MCD" and r["option_type"] == "call"
        ]
        assert len(mcd_call_rolls) == 0, (
            "MCD has only puts in the fixture but appears as a call roll — "
            "OPRA `str.contains('C')` bug regression"
        )


# ============================================================================
# New tool: opex_concentration
# ============================================================================


@pytest.mark.asyncio
class TestOpexConcentration:
    async def test_returns_ranked_results(self, oi_parquet: Path):
        from servers.oi_changes import call_tool

        result = await call_tool("opex_concentration", {})
        payload = json.loads(result[0].text)
        assert "results" in payload
        results = payload["results"]
        assert len(results) > 0

        # Each result must include the documented schema fields
        first = results[0]
        for field in ("ticker", "concentrated_expiry", "concentration_pct", "total_oi"):
            assert field in first, f"Missing field {field} in result"

        # Sorted descending by concentration
        pct_values = [r["concentration_pct"] for r in results]
        assert pct_values == sorted(pct_values, reverse=True)

    async def test_aapl_has_high_concentration(self, oi_parquet: Path):
        """The synthetic AAPL fixture has 8 contracts on the 2026-03-20 expiry
        and nothing on other expiries → 100% concentration on that expiry."""
        from servers.oi_changes import call_tool

        result = await call_tool("opex_concentration", {"top_n": 10})
        payload = json.loads(result[0].text)
        results = payload["results"]
        aapl = next((r for r in results if r["ticker"] == "AAPL"), None)
        assert aapl is not None
        assert aapl["concentration_pct"] >= 90.0
        assert aapl["concentrated_expiry"] == "2026-03-20"

    async def test_pin_distance_for_near_spot_tickers(self, oi_parquet: Path):
        """When spot is within 5% of the highest-OI strike, include pin_distance."""
        from servers.oi_changes import call_tool

        result = await call_tool("opex_concentration", {"top_n": 10})
        payload = json.loads(result[0].text)
        results = payload["results"]
        # AAPL spot=162, max-OI strike near 160 → within 5%
        aapl = next((r for r in results if r["ticker"] == "AAPL"), None)
        assert aapl is not None
        assert "pin_distance_pct" in aapl
        assert aapl["pin_distance_pct"] is not None

    async def test_min_concentration_filter(self, oi_parquet: Path):
        from servers.oi_changes import call_tool

        result = await call_tool(
            "opex_concentration",
            {"min_concentration_pct": 95},
        )
        payload = json.loads(result[0].text)
        for r in payload["results"]:
            assert r["concentration_pct"] >= 95.0

    async def test_top_n_limits_results(self, oi_parquet: Path):
        from servers.oi_changes import call_tool

        result = await call_tool("opex_concentration", {"top_n": 1})
        payload = json.loads(result[0].text)
        assert len(payload["results"]) <= 1

    async def test_caveat_field_present(self, oi_parquet: Path):
        """Per audit & PLAN.md: tools that aggregate single-day data should
        document their scope so users don't over-interpret."""
        from servers.oi_changes import call_tool

        result = await call_tool("opex_concentration", {})
        payload = json.loads(result[0].text)
        # caveat or note describing single-day scope
        assert "caveat" in payload or "note" in payload

    async def test_min_total_oi_filters_low_volume_tickers(self, oi_parquet: Path):
        """Tickers with total OI below min_total_oi must be excluded."""
        from servers.oi_changes import call_tool

        # KC has only ~1100 total OI in the fixture; raise the floor above that
        result = await call_tool(
            "opex_concentration",
            {"min_total_oi": 5000},
        )
        payload = json.loads(result[0].text)
        tickers = [r["ticker"] for r in payload["results"]]
        assert "KC" not in tickers

    async def test_invalid_top_n_raises(self):
        from servers.oi_changes import call_tool

        with pytest.raises(ValueError, match="top_n"):
            await call_tool("opex_concentration", {"top_n": -1})

    async def test_invalid_concentration_pct_raises(self):
        from servers.oi_changes import call_tool

        with pytest.raises(ValueError, match="min_concentration_pct"):
            await call_tool("opex_concentration", {"min_concentration_pct": 150})
