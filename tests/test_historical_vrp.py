"""Tests for the volatility_risk_premium (VRP) tool — Phase 3 audit §3.6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import synthesise

pytestmark = pytest.mark.unit


@pytest.fixture
def screener_with_iv(stocks_dir: Path):
    """Single-day screener with deterministic IV30d for a few tickers."""
    df = synthesise.build_screener_df_for_date(
        "2026-03-17",
        iv30d_per_ticker={"AAPL": 0.30, "TSLA": 0.50, "SPY": 0.18},
    )
    synthesise.write_screener_parquet(stocks_dir, "2026-03-17", df)
    return stocks_dir


@pytest.fixture
def mock_yfinance_high_realised(monkeypatch):
    """Patch yfinance.Ticker so VRP test gets a realized vol HIGHER than IV → negative VRP."""
    class FakeTicker:
        def __init__(self, _symbol):
            pass

        def history(self, *_args, **_kwargs):
            # Fabricate price returns ~3% daily → realized σ_annual ≈ 0.03 * sqrt(252) ≈ 0.476
            rng = np.random.default_rng(123)
            prices = [100.0]
            for _ in range(60):
                prices.append(prices[-1] * (1 + rng.normal(0, 0.03)))
            return pd.DataFrame({"Close": prices})

    monkeypatch.setattr("servers.historical.Ticker", FakeTicker, raising=False)
    monkeypatch.setattr("yfinance.Ticker", FakeTicker)


@pytest.fixture
def mock_yfinance_low_realised(monkeypatch):
    """Realized vol ≈ 0.01*sqrt(252) ≈ 0.16 — well below typical IV → positive VRP."""
    class FakeTicker:
        def __init__(self, _symbol):
            pass

        def history(self, *_args, **_kwargs):
            rng = np.random.default_rng(7)
            prices = [100.0]
            for _ in range(60):
                prices.append(prices[-1] * (1 + rng.normal(0, 0.01)))
            return pd.DataFrame({"Close": prices})

    monkeypatch.setattr("servers.historical.Ticker", FakeTicker, raising=False)
    monkeypatch.setattr("yfinance.Ticker", FakeTicker)


@pytest.fixture
def mock_yfinance_unavailable(monkeypatch):
    """Patch yfinance to raise on .history() — simulating a yfinance outage."""
    class FakeTicker:
        def __init__(self, _symbol):
            pass

        def history(self, *_args, **_kwargs):
            raise RuntimeError("yfinance unavailable")

    monkeypatch.setattr("servers.historical.Ticker", FakeTicker, raising=False)
    monkeypatch.setattr("yfinance.Ticker", FakeTicker)


@pytest.mark.asyncio
class TestVolatilityRiskPremium:
    async def test_high_iv_low_realised_yields_positive_vrp(
        self, screener_with_iv, mock_yfinance_low_realised
    ):
        from servers.historical import call_tool

        # AAPL IV=0.30, realised ≈ 0.16 → VRP ≈ +0.14 → premium-selling
        result = await call_tool("volatility_risk_premium", {"symbol": "AAPL"})
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "AAPL"
        assert payload["vrp"] > 0
        assert payload["regime"] == "PREMIUM_SELLING"

    async def test_low_iv_high_realised_yields_negative_vrp(
        self, screener_with_iv, mock_yfinance_high_realised
    ):
        from servers.historical import call_tool

        # SPY IV=0.18, realised ≈ 0.48 → VRP ≈ -0.30 → premium-buying
        result = await call_tool("volatility_risk_premium", {"symbol": "SPY"})
        payload = json.loads(result[0].text)
        assert payload["vrp"] < 0
        assert payload["regime"] == "PREMIUM_BUYING"

    async def test_yfinance_unavailable_returns_error(
        self, screener_with_iv, mock_yfinance_unavailable
    ):
        from servers.historical import call_tool

        result = await call_tool("volatility_risk_premium", {"symbol": "AAPL"})
        payload = json.loads(result[0].text)
        assert "error" in payload
        # Must not raise — graceful degradation
        assert payload["error"]

    async def test_missing_symbol_raises(self):
        from servers.historical import call_tool

        with pytest.raises(ValueError, match="symbol"):
            await call_tool("volatility_risk_premium", {})

    async def test_unknown_ticker_returns_note(
        self, screener_with_iv, mock_yfinance_low_realised
    ):
        """Ticker not in screener → no IV → cannot compute VRP."""
        from servers.historical import call_tool

        result = await call_tool("volatility_risk_premium", {"symbol": "ZZZZ"})
        payload = json.loads(result[0].text)
        assert "error" in payload or "note" in payload

    async def test_aggregated_no_raw_prices(
        self, screener_with_iv, mock_yfinance_low_realised
    ):
        """Per financial-security.md: VRP returns scalars only, no price series.
        Asserting an exact key set guards against future regressions."""
        from servers.historical import call_tool

        result = await call_tool("volatility_risk_premium", {"symbol": "AAPL"})
        payload = json.loads(result[0].text)
        expected = {
            "symbol", "iv30d", "realised_vol", "realised_window_days",
            "vrp", "regime", "interpretation",
        }
        assert set(payload.keys()) == expected

    async def test_iv_as_percentage_normalised(
        self, stocks_dir, mock_yfinance_low_realised
    ):
        """UW sometimes exports IV as 30.0 instead of 0.30. Heuristic > 5.0
        normalises by /100 so VRP doesn't blow up."""
        from servers.historical import call_tool

        df = synthesise.build_screener_df_for_date(
            "2026-03-17",
            iv30d_per_ticker={"WEIRD": 30.0},  # would-be 3000% if interpreted as decimal
        )
        synthesise.write_screener_parquet(stocks_dir, "2026-03-17", df)

        result = await call_tool("volatility_risk_premium", {"symbol": "WEIRD"})
        payload = json.loads(result[0].text)
        # Normalised iv30d is 0.30 — VRP should be in a plausible range, not ~30
        assert -1.0 < payload["vrp"] < 1.0
        assert payload["iv30d"] == 0.30
