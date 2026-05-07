"""Tests for servers.options_structure — Phase 2 new server.

Covers:
  - New tool: dealer_delta_exposure (DEX) — audit §3.1
  - Moved tools (smoke): iv_term_structure, term_skew, front_end_iv_ratio,
    today_gamma_flip, gamma_exposure_profile (now with dte_max)
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fixtures import synthesise

pytestmark = pytest.mark.unit


def _opra(root: str, expiry: date, cp: str, strike: float) -> str:
    return f"{root}{expiry.strftime('%y%m%d')}{cp}{int(strike * 1000):08d}"


def _build_options_df(
    *,
    symbol: str = "SPY",
    expiry: date | None = None,
    spot: float = 100.0,
    call_oi_per_strike: dict[float, int] | None = None,
    put_oi_per_strike: dict[float, int] | None = None,
    iv: float = 0.20,
) -> pd.DataFrame:
    """Build a minimal options DataFrame with the columns options_structure needs."""
    if expiry is None:
        expiry = date.today() + timedelta(days=14)
    rows: list[dict] = []
    for strike, oi in (call_oi_per_strike or {}).items():
        delta = max(0.05, min(0.95, 0.5 + (spot - strike) / spot))
        rows.append({
            "option_symbol": _opra(symbol, expiry, "C", strike),
            "underlying_symbol": symbol,
            "option_type": "call",
            "strike": strike,
            "expiry": expiry.strftime("%Y-%m-%d"),
            "underlying_price": spot,
            "delta": delta,
            "gamma": 0.01,
            "vega": 0.10,
            "theta": -0.05,
            "implied_volatility": iv,
            "open_interest": oi,
            "size": 100,
            "premium": 1000.0,
            "price": 5.0,
            "side": "ask",
            "executed_at": "2026-03-17 10:00:00",
            "sector": "Index",
            "volume": 100,
        })
    for strike, oi in (put_oi_per_strike or {}).items():
        delta = -max(0.05, min(0.95, 0.5 - (spot - strike) / spot))
        rows.append({
            "option_symbol": _opra(symbol, expiry, "P", strike),
            "underlying_symbol": symbol,
            "option_type": "put",
            "strike": strike,
            "expiry": expiry.strftime("%Y-%m-%d"),
            "underlying_price": spot,
            "delta": delta,
            "gamma": 0.01,
            "vega": 0.10,
            "theta": -0.05,
            "implied_volatility": iv,
            "open_interest": oi,
            "size": 100,
            "premium": 1000.0,
            "price": 5.0,
            "side": "ask",
            "executed_at": "2026-03-17 10:00:00",
            "sector": "Index",
            "volume": 100,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def options_parquet_call_heavy(stocks_dir: Path) -> Path:
    """Call-heavy SPY (2:1 call OI vs put OI) — DEX should be positive."""
    df = _build_options_df(
        symbol="SPY",
        spot=100.0,
        call_oi_per_strike={100.0: 2000, 105.0: 2000, 95.0: 2000},
        put_oi_per_strike={95.0: 1000, 100.0: 1000, 105.0: 1000},
    )
    return synthesise.write_options_parquet(stocks_dir, "2026-03-17", df)


@pytest.fixture
def options_parquet_put_heavy(stocks_dir: Path) -> Path:
    """Put-heavy QQQ (2:1 put OI vs call OI) — DEX should be negative."""
    df = _build_options_df(
        symbol="QQQ",
        spot=100.0,
        call_oi_per_strike={100.0: 1000, 105.0: 1000},
        put_oi_per_strike={95.0: 2000, 100.0: 2000, 105.0: 2000},
    )
    return synthesise.write_options_parquet(stocks_dir, "2026-03-17", df)


@pytest.fixture
def options_parquet_with_term_structure(stocks_dir: Path) -> Path:
    """SPY with two expiries and different IV — for term_structure / skew / DTE tests."""
    near = date.today() + timedelta(days=7)
    far = date.today() + timedelta(days=60)

    near_df = _build_options_df(
        symbol="SPY",
        expiry=near,
        spot=100.0,
        call_oi_per_strike={100.0: 1000, 105.0: 500},
        put_oi_per_strike={95.0: 500, 100.0: 1000},
        iv=0.30,  # high front-month IV
    )
    far_df = _build_options_df(
        symbol="SPY",
        expiry=far,
        spot=100.0,
        call_oi_per_strike={100.0: 500, 110.0: 300},
        put_oi_per_strike={90.0: 300, 100.0: 500},
        iv=0.18,  # lower back-month IV → backwardation
    )
    df = pd.concat([near_df, far_df], ignore_index=True)
    return synthesise.write_options_parquet(stocks_dir, "2026-03-17", df)


# ============================================================================
# dealer_delta_exposure (NEW)
# ============================================================================


@pytest.mark.asyncio
class TestDealerDeltaExposure:
    async def test_call_heavy_returns_positive_dex(self, options_parquet_call_heavy: Path):
        from servers.options_structure import call_tool

        result = await call_tool("dealer_delta_exposure", {"symbol": "SPY"})
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "SPY"
        assert payload["net_dex"] > 0, "Call-heavy OI must yield positive net DEX"
        assert payload["call_dex"] > 0
        assert payload["put_dex"] < 0  # puts have negative delta

    async def test_put_heavy_returns_negative_dex(self, options_parquet_put_heavy: Path):
        from servers.options_structure import call_tool

        result = await call_tool("dealer_delta_exposure", {"symbol": "QQQ"})
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "QQQ"
        assert payload["net_dex"] < 0, "Put-heavy OI must yield negative net DEX"

    async def test_dte_filter_excludes_leaps(self, stocks_dir: Path):
        """LEAPs should be excluded by default dte_max=45. The audit notes
        DEX should focus on near-term hedging just like GEX."""
        from servers.options_structure import call_tool

        near = date.today() + timedelta(days=7)
        leap = date.today() + timedelta(days=400)
        # Near calls only, LEAP puts only
        near_df = _build_options_df(
            symbol="ABC",
            expiry=near,
            spot=100.0,
            call_oi_per_strike={100.0: 1000},
        )
        leap_df = _build_options_df(
            symbol="ABC",
            expiry=leap,
            spot=100.0,
            put_oi_per_strike={100.0: 100_000},
        )
        df = pd.concat([near_df, leap_df], ignore_index=True)
        synthesise.write_options_parquet(stocks_dir, "2026-03-17", df)

        # Default dte_max=45 → LEAP put filtered out → DEX is positive (calls only)
        result = await call_tool("dealer_delta_exposure", {"symbol": "ABC"})
        payload = json.loads(result[0].text)
        assert payload["net_dex"] > 0

    async def test_dte_max_param_overrides_default(self, stocks_dir: Path):
        """Setting dte_max=None includes everything → DEX flips negative."""
        from servers.options_structure import call_tool

        near = date.today() + timedelta(days=7)
        leap = date.today() + timedelta(days=400)
        near_df = _build_options_df(symbol="ABC", expiry=near, call_oi_per_strike={100.0: 1000})
        leap_df = _build_options_df(symbol="ABC", expiry=leap, put_oi_per_strike={100.0: 100_000})
        df = pd.concat([near_df, leap_df], ignore_index=True)
        synthesise.write_options_parquet(stocks_dir, "2026-03-17", df)

        result = await call_tool("dealer_delta_exposure", {"symbol": "ABC", "dte_max": 500})
        payload = json.loads(result[0].text)
        assert payload["net_dex"] < 0  # LEAP puts dominate

    async def test_missing_symbol_raises(self):
        from servers.options_structure import call_tool

        with pytest.raises(ValueError, match="symbol"):
            await call_tool("dealer_delta_exposure", {})

    async def test_no_data_returns_error(self, stocks_dir: Path):
        """Empty data dir → no Parquet → error."""
        from servers.options_structure import call_tool

        result = await call_tool("dealer_delta_exposure", {"symbol": "AAPL"})
        payload = json.loads(result[0].text)
        assert "error" in payload

    async def test_aggregated_no_raw_trades(self, options_parquet_call_heavy: Path):
        """Per financial-security.md: no executed_at, no per-trade timestamps."""
        from servers.options_structure import call_tool

        result = await call_tool("dealer_delta_exposure", {"symbol": "SPY"})
        text = result[0].text
        assert "executed_at" not in text
        assert "10:00:00" not in text


# ============================================================================
# Moved tools — smoke tests (full coverage stays in their original tests)
# ============================================================================


@pytest.mark.asyncio
class TestMovedTools:
    async def test_iv_term_structure_dispatches(self, options_parquet_with_term_structure: Path):
        from servers.options_structure import call_tool

        result = await call_tool("iv_term_structure", {"symbol": "SPY"})
        payload = json.loads(result[0].text)
        assert "structure" in payload
        # Front IV (0.30) > back IV (0.18) → backwardation
        assert payload["structure"] == "BACKWARDATION"

    async def test_today_gamma_flip_dispatches(self, options_parquet_with_term_structure: Path):
        """Even with two expiries, today_gamma_flip must filter to the nearest
        and produce a valid result without crashing."""
        from servers.options_structure import call_tool

        result = await call_tool("today_gamma_flip", {"symbol": "SPY"})
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "SPY"
        # Regime is binary (POSITIVE/NEGATIVE) for this tool — preserves the
        # original options_flow.py contract; FULLY_* / FLAT belong to gamma_exposure_profile.
        assert payload["regime"] in {"POSITIVE", "NEGATIVE"}

    async def test_gamma_exposure_profile_with_dte_max(self, options_parquet_with_term_structure: Path):
        """The new dte_max parameter must be respected — different DTE windows
        produce different per-strike rows (the LEAP strikes are extra)."""
        from servers.options_structure import call_tool

        # Default dte_max=45 → only near (7 DTE) expiry's strikes
        result_default = await call_tool("gamma_exposure_profile", {"symbol": "SPY"})
        payload_default = json.loads(result_default[0].text)

        # dte_max=365 → both expiries → strictly more strikes
        result_all = await call_tool("gamma_exposure_profile", {"symbol": "SPY", "dte_max": 365})
        payload_all = json.loads(result_all[0].text)

        # Per-strike output reflects the included expiries
        assert len(payload_all["per_strike"]) > len(payload_default["per_strike"])
        # Default reports the dte_max it used
        assert payload_default["dte_max"] == 45

    async def test_term_skew_dispatches(self, options_parquet_with_term_structure: Path):
        from servers.options_structure import call_tool

        result = await call_tool(
            "term_skew",
            {"symbol": "SPY", "dte_target": 7},
        )
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "SPY"

    async def test_front_end_iv_ratio_dispatches(self, options_parquet_with_term_structure: Path):
        from servers.options_structure import call_tool

        result = await call_tool(
            "front_end_iv_ratio",
            {"symbol": "SPY", "near_dte": 7, "far_dte": 60},
        )
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "SPY"
        # Front IV 0.30 / back IV 0.18 ≈ 1.67 → BACKWARDATION
        assert payload["regime"] == "BACKWARDATION"

    async def test_front_end_iv_ratio_rejects_equal_dtes(self):
        from servers.options_structure import call_tool

        with pytest.raises(ValueError, match="near_dte"):
            await call_tool("front_end_iv_ratio", {"symbol": "SPY", "near_dte": 30, "far_dte": 30})


@pytest.mark.asyncio
async def test_server_lists_seven_tools():
    """Acceptance: uw-options-structure exposes 5 moved + DEX + vanna_charm = 7."""
    from servers.options_structure import list_tools

    tools = await list_tools()
    names = {t.name for t in tools}
    expected = {
        "iv_term_structure",
        "term_skew",
        "front_end_iv_ratio",
        "today_gamma_flip",
        "gamma_exposure_profile",
        "dealer_delta_exposure",
        "vanna_charm_exposure",
    }
    assert names == expected


# ============================================================================
# vanna_charm_exposure (NEW Phase 4)
# ============================================================================


@pytest.mark.asyncio
class TestVannaCharmExposure:
    async def test_call_heavy_oi_yields_negative_vanna(
        self, options_parquet_call_heavy: Path
    ):
        """Per-row vanna = -vega × delta / (σ × S). Calls: delta > 0 → vanna < 0.
        Heavy call OI → public net vanna negative (dealer-vanna positive)."""
        from servers.options_structure import call_tool

        result = await call_tool("vanna_charm_exposure", {"symbol": "SPY"})
        payload = json.loads(result[0].text)
        assert payload["symbol"] == "SPY"
        assert payload["net_vanna"] < 0

    async def test_put_heavy_oi_yields_positive_vanna(
        self, options_parquet_put_heavy: Path
    ):
        """Heavy put OI → put delta < 0 → per-row vanna > 0 → public net positive."""
        from servers.options_structure import call_tool

        result = await call_tool("vanna_charm_exposure", {"symbol": "QQQ"})
        payload = json.loads(result[0].text)
        assert payload["net_vanna"] > 0

    async def test_returns_charm_field(self, options_parquet_call_heavy: Path):
        from servers.options_structure import call_tool

        result = await call_tool("vanna_charm_exposure", {"symbol": "SPY"})
        payload = json.loads(result[0].text)
        assert "net_charm" in payload

    async def test_call_heavy_charm_sign(self, options_parquet_call_heavy: Path):
        """Call-heavy book with delta>0, theta<0, price>0 →
        charm = -theta × delta / price > 0 (positive net charm)."""
        from servers.options_structure import call_tool

        result = await call_tool("vanna_charm_exposure", {"symbol": "SPY"})
        payload = json.loads(result[0].text)
        assert payload["net_charm"] > 0

    async def test_invalid_dte_max_raises(self):
        from servers.options_structure import call_tool

        with pytest.raises(ValueError, match="dte_max"):
            await call_tool("vanna_charm_exposure", {"symbol": "SPY", "dte_max": 0})

    async def test_dte_filter_excludes_leaps(self, stocks_dir: Path):
        """LEAP positions filtered out by default (dte_max=45)."""
        from datetime import date, timedelta
        from servers.options_structure import call_tool

        near = date.today() + timedelta(days=7)
        leap = date.today() + timedelta(days=400)
        near_df = _build_options_df(symbol="ABC", expiry=near, call_oi_per_strike={100.0: 1000})
        leap_df = _build_options_df(symbol="ABC", expiry=leap, put_oi_per_strike={100.0: 100_000})
        df = pd.concat([near_df, leap_df], ignore_index=True)
        synthesise.write_options_parquet(stocks_dir, "2026-03-17", df)

        # Default (LEAPs excluded) → only the call → vanna < 0
        result = await call_tool("vanna_charm_exposure", {"symbol": "ABC"})
        payload = json.loads(result[0].text)
        assert payload["net_vanna"] < 0

        # dte_max=500 (LEAPs included) → put dominates → vanna flips
        result_all = await call_tool("vanna_charm_exposure", {"symbol": "ABC", "dte_max": 500})
        payload_all = json.loads(result_all[0].text)
        assert payload_all["net_vanna"] > 0

    async def test_missing_symbol_raises(self):
        from servers.options_structure import call_tool

        with pytest.raises(ValueError, match="symbol"):
            await call_tool("vanna_charm_exposure", {})

    async def test_no_data_returns_error(self, stocks_dir: Path):
        from servers.options_structure import call_tool

        result = await call_tool("vanna_charm_exposure", {"symbol": "AAPL"})
        payload = json.loads(result[0].text)
        assert "error" in payload

    async def test_aggregated_no_raw_trades(self, options_parquet_call_heavy: Path):
        from servers.options_structure import call_tool

        result = await call_tool("vanna_charm_exposure", {"symbol": "SPY"})
        text = result[0].text
        assert "executed_at" not in text
        assert "10:00:00" not in text
