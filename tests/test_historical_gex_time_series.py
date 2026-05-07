"""Tests for the new gex_time_series tool (Phase 3).

Multi-day Zero Gamma Level + total GEX trajectory for a symbol. Detects
regime-flip dates where ZGL crossed spot.
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


def _build_spy_day(
    file_date: str,
    near_expiry: date,
    *,
    spot: float,
) -> pd.DataFrame:
    """Build a SPY options DataFrame whose strike-level GEX profile yields
    ZGL ≈ 110 via linear interpolation between strikes 100 and 115.

    Math (per-row GEX = gamma × OI × 100 × spot):
        Put@100 OI=2000  → cum = -2000 × spot_factor
        Call@115 OI=3000 → cum = +1000 × spot_factor (sign flip)
    ZGL = 100 + 15 × (2000 / 3000) = 110.
    """
    rows = []
    # Put far OTM — drives cumulative negative
    rows.append({
        "option_symbol": _opra("SPY", near_expiry, "P", 100.0),
        "underlying_symbol": "SPY",
        "option_type": "put",
        "strike": 100.0,
        "expiry": near_expiry.strftime("%Y-%m-%d"),
        "underlying_price": spot,
        "delta": -0.4,
        "gamma": 0.01,
        "vega": 0.10,
        "theta": -0.05,
        "implied_volatility": 0.20,
        "open_interest": 2000,
        "size": 100,
        "premium": 1000.0,
        "price": 5.0,
        "side": "ask",
        "executed_at": "10:00:00",
        "sector": "Index",
        "volume": 100,
    })
    # Call slightly OTM — flips cumulative positive
    rows.append({
        "option_symbol": _opra("SPY", near_expiry, "C", 115.0),
        "underlying_symbol": "SPY",
        "option_type": "call",
        "strike": 115.0,
        "expiry": near_expiry.strftime("%Y-%m-%d"),
        "underlying_price": spot,
        "delta": 0.4,
        "gamma": 0.01,
        "vega": 0.10,
        "theta": -0.05,
        "implied_volatility": 0.20,
        "open_interest": 3000,
        "size": 100,
        "premium": 1000.0,
        "price": 5.0,
        "side": "ask",
        "executed_at": "10:00:00",
        "sector": "Index",
        "volume": 100,
    })
    return pd.DataFrame(rows)


@pytest.fixture
def gex_history(stocks_dir: Path):
    """Build 5 days of SPY options data with the spot crossing the ZGL on day 3.

    Day 1: spot=100, ZGL=110 → spot < ZGL → NEGATIVE
    Day 2: spot=105, ZGL=110 → NEGATIVE (still below)
    Day 3: spot=115, ZGL=110 → POSITIVE (regime flip!)
    Day 4: spot=120, ZGL=110 → POSITIVE
    Day 5: spot=118, ZGL=110 → POSITIVE
    """
    near_expiry = date.today() + timedelta(days=14)
    file_dates = [f"2026-03-{day:02d}" for day in (10, 11, 12, 13, 14)]
    spots = [100.0, 105.0, 115.0, 120.0, 118.0]
    for fd, spot in zip(file_dates, spots):
        df = _build_spy_day(fd, near_expiry, spot=spot)
        synthesise.write_options_parquet(stocks_dir, fd, df)
    return file_dates


@pytest.mark.asyncio
class TestGexTimeSeries:
    async def test_returns_per_date_zgl_trajectory(self, gex_history):
        from servers.historical import call_tool

        result = await call_tool("gex_time_series", {"symbol": "SPY", "days": 5})
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "SPY"
        assert "trajectory" in payload
        assert len(payload["trajectory"]) == 5
        for entry in payload["trajectory"]:
            assert "date" in entry
            assert "zero_gamma_level" in entry
            assert "total_gex" in entry
            assert "spot" in entry
            assert "regime" in entry

    async def test_detects_regime_flip(self, gex_history):
        """Spot crosses ZGL=110 between day 2 (spot=105) and day 3 (spot=115)."""
        from servers.historical import call_tool

        result = await call_tool("gex_time_series", {"symbol": "SPY", "days": 5})
        payload = json.loads(result[0].text)
        flips = payload.get("regime_flip_dates", [])
        # At least one flip should be reported around the spot crossing
        assert len(flips) >= 1

    async def test_missing_symbol_raises(self):
        from servers.historical import call_tool

        with pytest.raises(ValueError, match="symbol"):
            await call_tool("gex_time_series", {})

    async def test_no_data_returns_note(self, stocks_dir):
        from servers.historical import call_tool

        result = await call_tool("gex_time_series", {"symbol": "SPY"})
        payload = json.loads(result[0].text)
        assert "note" in payload or "error" in payload

    async def test_caching_returns_same_result(self, gex_history):
        """Two calls with the same args should return identical trajectories
        AND populate the per-process cache so subsequent runs avoid recomputing."""
        import servers.historical as hist
        from servers.historical import call_tool

        # Reset cache for deterministic assertion
        hist._GEX_DAILY_CACHE.clear()
        r1 = await call_tool("gex_time_series", {"symbol": "SPY", "days": 5})
        cache_size_after_first = len(hist._GEX_DAILY_CACHE)
        r2 = await call_tool("gex_time_series", {"symbol": "SPY", "days": 5})
        assert r1[0].text == r2[0].text
        # First call must have populated the cache; second call must not have grown it
        assert cache_size_after_first > 0
        assert len(hist._GEX_DAILY_CACHE) == cache_size_after_first

    async def test_invalid_days_raises(self):
        from servers.historical import call_tool

        with pytest.raises(ValueError, match="days"):
            await call_tool("gex_time_series", {"symbol": "SPY", "days": 0})

    async def test_aggregated_no_raw_trades(self, gex_history):
        from servers.historical import call_tool

        result = await call_tool("gex_time_series", {"symbol": "SPY", "days": 5})
        text = result[0].text
        assert "executed_at" not in text
        assert "10:00:00" not in text
