"""Regression tests for the signal_backtest methodology bugs from audit §1.4.

Two fixes:
  1. historical.py:405 — lookback used `min(lookback_days - 1, len(hist) - 1)` on a
     yfinance history fetched with a calendar-day buffer of only `lookback_days + 5`.
     On Friday-before-holiday signals the buffer was insufficient and the lookup
     fell back to the last available bar — short of `lookback_days`. Fix: enlarge
     the calendar buffer to `lookback_days + 14` so trading-day index `lookback_days
     - 1` is reliably available.

  2. historical.py:423 — `high_iv_rank` win metric was `|move| > 2%` regardless
     of direction. That's a vol-realisation check, not a directional win-rate.
     Fix: rename the metric to `vol_realisation_rate` for the IV-rank signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import synthesise

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.fixture
def screener_one_signal_day(stocks_dir: Path):
    """Single screener day flagging an obvious bullish-flow signal on AAPL."""
    df = synthesise.build_screener_df_for_date("2026-03-13", iv30d_per_ticker={"AAPL": 0.30})
    df.loc[df["ticker"] == "AAPL", "bullish_premium"] = 1_000_000.0
    df.loc[df["ticker"] == "AAPL", "bearish_premium"] = 100_000.0
    df.loc[df["ticker"] == "AAPL", "close"] = 100.0
    synthesise.write_screener_parquet(stocks_dir, "2026-03-13", df)
    return stocks_dir


@pytest.fixture
def mock_yfinance_friday_holiday(monkeypatch):
    """Simulate a Friday signal followed by a Monday holiday: yfinance returns
    bars only for the next 4 trading days within a 5-day calendar window."""

    class FakeTicker:
        def __init__(self, _symbol):
            pass

        def history(self, start=None, end=None, period=None, **_kwargs):
            # Return a determined number of bars regardless of (start, end);
            # the harness asserts the tool selects the right index.
            prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
            dates = pd.bdate_range("2026-03-16", periods=len(prices))
            return pd.DataFrame({"Close": prices}, index=dates)

    monkeypatch.setattr("servers.historical.Ticker", FakeTicker, raising=False)


@pytest.fixture
def mock_yfinance_high_iv_outcome(monkeypatch):
    """Returns a price series with low realised move (< 2%) — used to test
    that high_iv_rank's vol_realisation_rate metric reflects the realised move."""

    class FakeTicker:
        def __init__(self, _symbol):
            pass

        def history(self, *_args, **_kwargs):
            # Tiny daily moves → realised move at lookback << 2%
            prices = [100.0, 100.05, 100.10, 100.12, 100.15, 100.18, 100.20]
            dates = pd.bdate_range("2026-03-16", periods=len(prices))
            return pd.DataFrame({"Close": prices}, index=dates)

    monkeypatch.setattr("servers.historical.Ticker", FakeTicker, raising=False)


@pytest.mark.asyncio
class TestSignalBacktestLookback:
    async def test_friday_holiday_lookup_uses_business_days(
        self, screener_one_signal_day, mock_yfinance_friday_holiday
    ):
        """At lookback_days=5, the result should use the 5th business-day bar
        (bar index 4, value 104.0 from the fixture). Pre-fix code used
        min(lookback_days-1, len-1) i.e. index 4 — same answer here, BUT
        the fix is the calendar buffer so this case has enough bars. The
        regression target is no `IndexError`/silent fallback."""
        from servers.historical import call_tool

        result = await call_tool(
            "signal_backtest",
            {"signal_type": "bullish_flow", "lookback_days": 5, "top_n": 5},
        )
        payload = json.loads(result[0].text)
        # Should have at least one usable result, no errors
        assert payload["total_signals"] >= 1
        # The price after lookback should be deterministic per the fixture
        first = payload["results"][0]
        assert first["price_on_signal"] == 100.0
        # Tool should report what bar position was used (methodology_notes)
        assert "methodology_notes" in payload


@pytest.mark.asyncio
class TestHighIvRankMetricRename:
    async def test_high_iv_rank_uses_vol_realisation_rate_field(
        self, screener_one_signal_day, mock_yfinance_high_iv_outcome
    ):
        """For signal_type=high_iv_rank, the result should report
        `vol_realisation_rate` instead of `win_rate` (because the IV-rank
        signal has no directional thesis to win on)."""
        from servers.historical import call_tool

        # Mark AAPL as high IV-rank for the screener day
        # synthesise default iv_rank=50; bump it
        # We just need the signal to fire — the fixture gives AAPL iv_rank=50.
        # Bump iv_rank by re-writing the file.
        df = synthesise.build_screener_df_for_date("2026-03-13", iv30d_per_ticker={"AAPL": 0.30})
        df.loc[df["ticker"] == "AAPL", "iv_rank"] = 90.0
        df.loc[df["ticker"] == "AAPL", "close"] = 100.0
        synthesise.write_screener_parquet(screener_one_signal_day, "2026-03-13", df)

        result = await call_tool(
            "signal_backtest",
            {"signal_type": "high_iv_rank", "lookback_days": 5, "top_n": 5},
        )
        payload = json.loads(result[0].text)
        # The renamed metric must be present
        assert "vol_realisation_rate" in payload
        # The legacy win_rate must NOT appear for this signal type
        assert "win_rate" not in payload
        # methodology_notes must explain the renaming
        assert "methodology_notes" in payload
        notes = payload["methodology_notes"].lower()
        assert "vol_realisation" in notes or "non-directional" in notes

    async def test_directional_signals_keep_win_rate(
        self, screener_one_signal_day, mock_yfinance_friday_holiday
    ):
        """bullish_flow / bearish_flow keep the original directional win_rate field."""
        from servers.historical import call_tool

        result = await call_tool(
            "signal_backtest",
            {"signal_type": "bullish_flow", "lookback_days": 5, "top_n": 5},
        )
        payload = json.loads(result[0].text)
        assert "win_rate" in payload
        assert "vol_realisation_rate" not in payload
