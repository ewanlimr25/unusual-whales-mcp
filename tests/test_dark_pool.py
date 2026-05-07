"""Tests for servers.dark_pool — new tool dp_block_size_stratified.

Stratifies dark pool premium into 4 tiers — mega (>$10M), block (>$1M),
large (>$100k), retail (<$100k) — and reports buy/sell ratio per tier.
Implements audit §3.3 with citations to Easley-Lopez de Prado-O'Hara 2012.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
class TestDpBlockSizeStratified:
    async def test_returns_per_ticker_tier_breakdown(self, darkpool_parquet: Path):
        from servers.dark_pool import call_tool

        result = await call_tool("dp_block_size_stratified", {})
        payload = json.loads(result[0].text)
        assert "results" in payload
        results = payload["results"]
        assert len(results) > 0

        # Each ticker entry must contain the four tiers
        first = results[0]
        assert "ticker" in first
        for tier in ("mega", "block", "large", "retail"):
            assert tier in first, f"Missing tier {tier} in result"

        # Each tier must report buy_ratio and total_premium
        for tier in ("mega", "block", "large", "retail"):
            tier_data = first[tier]
            for field in ("buy_ratio", "total_premium", "trade_count"):
                assert field in tier_data, f"Missing {field} in {tier} tier"

    async def test_aapl_has_mega_tier_buy_pressure(self, darkpool_parquet: Path):
        """The fixture has a $12M AAPL trade above mid → mega tier is bullish."""
        from servers.dark_pool import call_tool

        result = await call_tool("dp_block_size_stratified", {"symbol": "AAPL"})
        payload = json.loads(result[0].text)
        results = payload["results"]
        aapl = next((r for r in results if r["ticker"] == "AAPL"), None)
        assert aapl is not None
        assert aapl["mega"]["trade_count"] >= 1
        # Mega trade is at price >= mid → buy_ratio ~= 1.0
        assert aapl["mega"]["buy_ratio"] >= 0.9

    async def test_tsla_block_tier_is_sell_dominated(self, darkpool_parquet: Path):
        """TSLA fixture: $8M and $1.5M block sells → block tier buy_ratio ~ 0."""
        from servers.dark_pool import call_tool

        result = await call_tool("dp_block_size_stratified", {"symbol": "TSLA"})
        payload = json.loads(result[0].text)
        results = payload["results"]
        tsla = next((r for r in results if r["ticker"] == "TSLA"), None)
        assert tsla is not None
        assert tsla["block"]["trade_count"] >= 2
        assert tsla["block"]["buy_ratio"] <= 0.1

    async def test_mcd_has_only_retail_activity(self, darkpool_parquet: Path):
        """MCD fixture has only retail-tier activity. Mega/block must be empty.
        Need to lower min_tier to retail to see retail-only tickers."""
        from servers.dark_pool import call_tool

        result = await call_tool(
            "dp_block_size_stratified",
            {"symbol": "MCD", "min_tier": "retail"},
        )
        payload = json.loads(result[0].text)
        results = payload["results"]
        mcd = next((r for r in results if r["ticker"] == "MCD"), None)
        assert mcd is not None
        assert mcd["mega"]["trade_count"] == 0
        assert mcd["block"]["trade_count"] == 0
        assert mcd["retail"]["trade_count"] > 0

    async def test_filter_min_tier_filters_results(self, darkpool_parquet: Path):
        """min_tier parameter excludes tickers without activity in the named
        tier or above. min_tier=block excludes MCD (retail-only)."""
        from servers.dark_pool import call_tool

        result = await call_tool(
            "dp_block_size_stratified",
            {"min_tier": "block"},
        )
        payload = json.loads(result[0].text)
        tickers = [r["ticker"] for r in payload["results"]]
        assert "MCD" not in tickers
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    async def test_top_n_limits_results(self, darkpool_parquet: Path):
        from servers.dark_pool import call_tool

        result = await call_tool("dp_block_size_stratified", {"top_n": 1})
        payload = json.loads(result[0].text)
        assert len(payload["results"]) <= 1

    async def test_invalid_min_tier_raises(self):
        from servers.dark_pool import call_tool

        with pytest.raises(ValueError, match="min_tier"):
            await call_tool("dp_block_size_stratified", {"min_tier": "garbage"})

    async def test_invalid_top_n_raises(self):
        from servers.dark_pool import call_tool

        with pytest.raises(ValueError, match="top_n"):
            await call_tool("dp_block_size_stratified", {"top_n": -1})

    async def test_aggregated_output_no_raw_trades(self, darkpool_parquet: Path):
        """Per financial-security.md: results must be aggregated, no raw trades.
        Output should not contain executed_at or per-trade prices."""
        from servers.dark_pool import call_tool

        result = await call_tool("dp_block_size_stratified", {})
        text = result[0].text
        assert "executed_at" not in text
        # No timestamp-like patterns
        assert "10:00:00" not in text
        assert "11:00:00" not in text

    async def test_symbol_case_insensitive(self, darkpool_parquet: Path):
        """Lowercase symbol input must resolve to uppercase ticker."""
        from servers.dark_pool import call_tool

        result = await call_tool(
            "dp_block_size_stratified",
            {"symbol": "aapl"},
        )
        payload = json.loads(result[0].text)
        results = payload["results"]
        assert any(r["ticker"] == "AAPL" for r in results)

    async def test_caveat_or_note_present(self, darkpool_parquet: Path):
        from servers.dark_pool import call_tool

        result = await call_tool("dp_block_size_stratified", {})
        payload = json.loads(result[0].text)
        assert "caveat" in payload or "note" in payload
