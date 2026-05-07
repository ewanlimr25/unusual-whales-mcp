"""Tests for the new iv_percentile_zscore tool (Phase 3).

Goyal-Saretto 2009 — IV percentile is a more robust sort than IV-rank
(which collapses on outliers). This tool computes the percentile rank of
each ticker's current IV30d within its trailing N-day distribution.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
class TestIvPercentileZscore:
    async def test_returns_per_ticker_percentile(self, screener_history):
        from servers.historical import call_tool

        result = await call_tool(
            "iv_percentile_zscore",
            {"symbol": "AAPL", "lookback_days": 29},
        )
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "AAPL"
        # AAPL IV is monotonically rising; latest reading should be at the top
        assert payload["iv_percentile"] >= 90.0
        assert "iv_zscore" in payload
        assert "current_iv30d" in payload

    async def test_outlier_robust_vs_iv_rank(self, screener_history):
        """TSLA has 29 readings of ~0.50 and one outlier of 5.0. IV-rank
        would compress the typical 0.50 reading to ~0%; percentile keeps it
        near the median."""
        from servers.historical import call_tool

        result = await call_tool(
            "iv_percentile_zscore",
            {"symbol": "TSLA", "lookback_days": 29},
        )
        payload = json.loads(result[0].text)
        # Latest TSLA IV is 0.50 (29 of 30 days are 0.50, one is the outlier)
        assert 30.0 <= payload["iv_percentile"] <= 70.0

    async def test_default_lookback_uses_all_available(self, screener_history):
        """Default lookback (252) just falls back to N-1 when fewer dates exist."""
        from servers.historical import call_tool

        result = await call_tool(
            "iv_percentile_zscore",
            {"symbol": "AAPL"},
        )
        payload = json.loads(result[0].text)
        assert payload["dates_used"] >= 5

    async def test_too_few_dates_returns_error(self, stocks_dir):
        """Empty stocks_dir → no data → graceful error."""
        from servers.historical import call_tool

        result = await call_tool("iv_percentile_zscore", {"symbol": "AAPL"})
        payload = json.loads(result[0].text)
        assert "error" in payload or "note" in payload

    async def test_missing_symbol_raises(self):
        from servers.historical import call_tool

        with pytest.raises(ValueError, match="symbol"):
            await call_tool("iv_percentile_zscore", {})

    async def test_zscore_classification(self, screener_history):
        """AAPL latest IV is the highest in the window → z-score positive."""
        from servers.historical import call_tool

        result = await call_tool(
            "iv_percentile_zscore",
            {"symbol": "AAPL", "lookback_days": 29},
        )
        payload = json.loads(result[0].text)
        assert payload["iv_zscore"] > 0
        assert payload["regime"] in {"HIGH_IV", "LOW_IV", "NORMAL"}
