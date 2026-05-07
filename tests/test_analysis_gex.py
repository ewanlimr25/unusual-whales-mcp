"""Tests for shared.analysis_gex — extracted GEX primitives.

Replaces the inline GEX computation duplicated in options_flow.py:556-625
(today_gamma_flip) and options_flow.py:748-819 (gamma_exposure_profile).

Adds optional `dte_max` parameter to address audit §2.2 LEAP-bias issue.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _opra_symbol(root: str, expiry: date, cp: str, strike: float) -> str:
    yymmdd = expiry.strftime("%y%m%d")
    return f"{root}{yymmdd}{cp}{int(strike * 1000):08d}"


def _make_gex_df(
    symbol: str = "SPY",
    expiry: date | None = None,
    *,
    call_strikes: list[tuple[float, float]] | None = None,
    put_strikes: list[tuple[float, float]] | None = None,
    spot: float = 100.0,
) -> pd.DataFrame:
    """Build a synthetic options DataFrame for GEX computation.

    Args:
        call_strikes / put_strikes: list of (strike, oi) tuples.
        Each contract is given gamma=0.01 (rough ATM value).
    """
    if expiry is None:
        expiry = date.today() + timedelta(days=7)
    rows: list[dict] = []
    for strike, oi in call_strikes or []:
        rows.append({
            "option_symbol": _opra_symbol(symbol, expiry, "C", strike),
            "underlying_symbol": symbol,
            "option_type": "call",
            "strike": strike,
            "gamma": 0.01,
            "open_interest": oi,
            "underlying_price": spot,
        })
    for strike, oi in put_strikes or []:
        rows.append({
            "option_symbol": _opra_symbol(symbol, expiry, "P", strike),
            "underlying_symbol": symbol,
            "option_type": "put",
            "strike": strike,
            "gamma": 0.01,
            "open_interest": oi,
            "underlying_price": spot,
        })
    return pd.DataFrame(rows)


# ============================================================================
# compute_gex_per_strike
# ============================================================================


class TestComputeGexPerStrike:
    def test_all_calls_positive_gex(self):
        from shared.analysis_gex import compute_gex_per_strike

        df = _make_gex_df(call_strikes=[(100.0, 1000), (105.0, 500)])
        result = compute_gex_per_strike(df)
        assert (result["gex"] > 0).all()
        # GEX = gamma * OI * 100 * spot = 0.01 * 1000 * 100 * 100 = 100_000
        assert result.loc[result["strike"] == 100.0, "gex"].iloc[0] == pytest.approx(100_000.0)

    def test_all_puts_negative_gex(self):
        from shared.analysis_gex import compute_gex_per_strike

        df = _make_gex_df(put_strikes=[(100.0, 1000), (95.0, 500)])
        result = compute_gex_per_strike(df)
        assert (result["gex"] < 0).all()

    def test_mixed_strikes_aggregated(self):
        """A strike with both calls and puts should have their GEX summed (calls
        positive, puts negative). When call OI > put OI at the same strike, net
        is positive."""
        from shared.analysis_gex import compute_gex_per_strike

        df = _make_gex_df(
            call_strikes=[(100.0, 2000)],  # +200_000
            put_strikes=[(100.0, 500)],    # -50_000
        )
        result = compute_gex_per_strike(df)
        assert len(result) == 1
        assert result["gex"].iloc[0] == pytest.approx(150_000.0)

    def test_does_not_mutate_input(self):
        from shared.analysis_gex import compute_gex_per_strike

        df = _make_gex_df(call_strikes=[(100.0, 1000)])
        original_cols = list(df.columns)
        _ = compute_gex_per_strike(df)
        assert list(df.columns) == original_cols
        assert "gex" not in df.columns

    def test_empty_dataframe_returns_empty(self):
        from shared.analysis_gex import compute_gex_per_strike

        df = pd.DataFrame(columns=[
            "option_symbol", "option_type", "strike", "gamma",
            "open_interest", "underlying_price",
        ])
        result = compute_gex_per_strike(df)
        assert len(result) == 0
        assert "strike" in result.columns
        assert "gex" in result.columns

    def test_filters_zero_or_missing_gamma(self):
        from shared.analysis_gex import compute_gex_per_strike

        df = _make_gex_df(call_strikes=[(100.0, 1000), (105.0, 500)])
        df.loc[1, "gamma"] = 0.0  # zero gamma should be filtered
        result = compute_gex_per_strike(df)
        # Only strike 100 survives
        assert len(result) == 1
        assert result["strike"].iloc[0] == 100.0

    def test_filters_zero_or_missing_oi(self):
        from shared.analysis_gex import compute_gex_per_strike

        df = _make_gex_df(call_strikes=[(100.0, 0), (105.0, 500)])
        result = compute_gex_per_strike(df)
        assert len(result) == 1
        assert result["strike"].iloc[0] == 105.0

    def test_dte_max_filter_excludes_leaps(self):
        """LEAPs (DTE > dte_max) must be filtered out. Audit §2.2: GEX is
        dominated by near-term gamma; LEAPs bias the ZGL."""
        from shared.analysis_gex import compute_gex_per_strike

        near_expiry = date.today() + timedelta(days=7)
        leap_expiry = date.today() + timedelta(days=365)
        rows: list[dict] = []
        for expiry, strike, oi in [(near_expiry, 100.0, 1000), (leap_expiry, 200.0, 1000)]:
            rows.append({
                "option_symbol": _opra_symbol("SPY", expiry, "C", strike),
                "option_type": "call",
                "strike": strike,
                "gamma": 0.01,
                "open_interest": oi,
                "underlying_price": 100.0,
            })
        df = pd.DataFrame(rows)
        result = compute_gex_per_strike(df, dte_max=45)
        strikes_kept = set(result["strike"].tolist())
        assert 100.0 in strikes_kept  # near-term
        assert 200.0 not in strikes_kept  # LEAP filtered out

    def test_dte_max_none_keeps_all(self):
        from shared.analysis_gex import compute_gex_per_strike

        near = date.today() + timedelta(days=7)
        leap = date.today() + timedelta(days=365)
        rows = [
            {
                "option_symbol": _opra_symbol("SPY", near, "C", 100.0),
                "option_type": "call", "strike": 100.0, "gamma": 0.01,
                "open_interest": 1000, "underlying_price": 100.0,
            },
            {
                "option_symbol": _opra_symbol("SPY", leap, "C", 200.0),
                "option_type": "call", "strike": 200.0, "gamma": 0.01,
                "open_interest": 1000, "underlying_price": 100.0,
            },
        ]
        df = pd.DataFrame(rows)
        result = compute_gex_per_strike(df, dte_max=None)
        assert len(result) == 2

    def test_dte_max_changes_zgl_on_leap_biased_data(self):
        """When LEAP put OI on a far OTM strike pulls cumulative GEX deeper
        negative, applying dte_max should produce a different (closer-in) ZGL.
        """
        from shared.analysis_gex import compute_gex_per_strike, compute_zero_gamma_level

        near = date.today() + timedelta(days=7)
        leap = date.today() + timedelta(days=365)
        rows = [
            # Near-term: small puts at 95 (gex=-10k), big calls at 105 (gex=+30k)
            # Without LEAP: ZGL ≈ 95 + 10 * (10/30) ≈ 98.33
            {"option_symbol": _opra_symbol("SPY", near, "P", 95.0),
             "option_type": "put", "strike": 95.0, "gamma": 0.01,
             "open_interest": 100, "underlying_price": 100.0},
            {"option_symbol": _opra_symbol("SPY", near, "C", 105.0),
             "option_type": "call", "strike": 105.0, "gamma": 0.01,
             "open_interest": 300, "underlying_price": 100.0},
            # LEAP put at far OTM strike 50 → biases the ZGL upward when included
            # With LEAP: ZGL ≈ 95 + 10 * (20/30) ≈ 101.67
            {"option_symbol": _opra_symbol("SPY", leap, "P", 50.0),
             "option_type": "put", "strike": 50.0, "gamma": 0.01,
             "open_interest": 100, "underlying_price": 100.0},
        ]
        df = pd.DataFrame(rows)

        unfiltered = compute_gex_per_strike(df, dte_max=None)
        filtered = compute_gex_per_strike(df, dte_max=45)
        zgl_unfiltered = compute_zero_gamma_level(unfiltered)
        zgl_filtered = compute_zero_gamma_level(filtered)
        # Both must produce a ZGL, and they must differ
        assert zgl_unfiltered is not None
        assert zgl_filtered is not None
        assert abs(zgl_unfiltered - zgl_filtered) > 1.0


# ============================================================================
# compute_zero_gamma_level
# ============================================================================


class TestComputeZeroGammaLevel:
    def test_returns_none_when_all_positive(self):
        from shared.analysis_gex import compute_zero_gamma_level

        strike_gex = pd.DataFrame({
            "strike": [95.0, 100.0, 105.0],
            "gex": [10.0, 20.0, 30.0],
        })
        assert compute_zero_gamma_level(strike_gex) is None

    def test_returns_none_when_all_negative(self):
        from shared.analysis_gex import compute_zero_gamma_level

        strike_gex = pd.DataFrame({
            "strike": [95.0, 100.0, 105.0],
            "gex": [-10.0, -20.0, -30.0],
        })
        assert compute_zero_gamma_level(strike_gex) is None

    def test_finds_flip_via_linear_interpolation(self):
        from shared.analysis_gex import compute_zero_gamma_level

        # Cumulative GEX flips from negative to positive between strikes 100 and 105
        # Strike 95: gex=-30 → cum=-30
        # Strike 100: gex=+10 → cum=-20
        # Strike 105: gex=+30 → cum=+10  ← flip happens between 100 and 105
        strike_gex = pd.DataFrame({
            "strike": [95.0, 100.0, 105.0],
            "gex": [-30.0, 10.0, 30.0],
        })
        zgl = compute_zero_gamma_level(strike_gex)
        # Linear interpolation: 100 + (105-100) * (20 / (20+10)) = ~103.33
        assert zgl is not None
        assert 100.0 < zgl < 105.0

    def test_empty_returns_none(self):
        from shared.analysis_gex import compute_zero_gamma_level

        empty = pd.DataFrame({"strike": [], "gex": []})
        assert compute_zero_gamma_level(empty) is None

    def test_exact_zero_crossing_returned(self):
        """Cumulative GEX of exactly 0 at a strike should return that strike,
        not silently fall through to None due to negative-zero arithmetic."""
        from shared.analysis_gex import compute_zero_gamma_level

        # cum: -30, 0, +10 — crossing lands precisely on strike 100
        strike_gex = pd.DataFrame({
            "strike": [95.0, 100.0, 105.0],
            "gex": [-30.0, 30.0, 10.0],
        })
        assert compute_zero_gamma_level(strike_gex) == 100.0

    def test_returns_first_of_multiple_crossings(self):
        """Bimodal distributions can have multiple sign flips — first wins."""
        from shared.analysis_gex import compute_zero_gamma_level

        # cum: -10, +10, -10, +20 → flips at strikes 100, 105, 110
        # First crossing is between 95 and 100; ZGL ≈ 95 + 5 * (10/20) = 97.5
        strike_gex = pd.DataFrame({
            "strike": [95.0, 100.0, 105.0, 110.0],
            "gex": [-10.0, 20.0, -20.0, 30.0],
        })
        zgl = compute_zero_gamma_level(strike_gex)
        assert zgl is not None
        assert 95.0 < zgl <= 100.0


# ============================================================================
# classify_gex_regime
# ============================================================================


class TestClassifyGexRegime:
    def test_positive_when_spot_above_zgl(self):
        from shared.analysis_gex import classify_gex_regime

        regime = classify_gex_regime(total_gex=1000.0, spot=105.0, zero_gamma_level=100.0)
        assert regime == "POSITIVE"

    def test_negative_when_spot_below_zgl(self):
        from shared.analysis_gex import classify_gex_regime

        regime = classify_gex_regime(total_gex=-1000.0, spot=95.0, zero_gamma_level=100.0)
        assert regime == "NEGATIVE"

    def test_fully_positive_when_no_zgl_and_total_positive(self):
        from shared.analysis_gex import classify_gex_regime

        regime = classify_gex_regime(total_gex=1000.0, spot=100.0, zero_gamma_level=None)
        assert regime == "FULLY_POSITIVE"

    def test_fully_negative_when_no_zgl_and_total_negative(self):
        from shared.analysis_gex import classify_gex_regime

        regime = classify_gex_regime(total_gex=-1000.0, spot=100.0, zero_gamma_level=None)
        assert regime == "FULLY_NEGATIVE"

    def test_flat_when_no_zgl_and_total_exactly_zero(self):
        from shared.analysis_gex import classify_gex_regime

        regime = classify_gex_regime(total_gex=0.0, spot=100.0, zero_gamma_level=None)
        assert regime == "FLAT"


class TestComputeGexPerStrikeErrorPaths:
    def test_raises_when_dte_set_but_no_expiry_or_symbol(self):
        from shared.analysis_gex import compute_gex_per_strike

        # Has gamma/OI but no option_symbol or expiry column
        df = pd.DataFrame({
            "option_type": ["call"],
            "strike": [100.0],
            "gamma": [0.01],
            "open_interest": [1000],
            "underlying_price": [100.0],
        })
        with pytest.raises(ValueError, match="expiry"):
            compute_gex_per_strike(df, dte_max=45)

    def test_dte_set_with_no_symbol_but_keeps_all_when_dte_none(self):
        from shared.analysis_gex import compute_gex_per_strike

        df = pd.DataFrame({
            "option_type": ["call"],
            "strike": [100.0],
            "gamma": [0.01],
            "open_interest": [1000],
            "underlying_price": [100.0],
        })
        # dte_max=None bypasses the expiry requirement
        result = compute_gex_per_strike(df, dte_max=None)
        assert len(result) == 1

    def test_all_filtered_returns_empty_schema(self):
        """When the gamma/OI validity filter strips every row, output is
        empty but schema-stable (callers can index columns safely)."""
        from shared.analysis_gex import compute_gex_per_strike

        df = _make_gex_df(call_strikes=[(100.0, 1000)])
        df.loc[0, "gamma"] = 0.0  # filter removes the only row
        result = compute_gex_per_strike(df)
        assert len(result) == 0
        assert "strike" in result.columns
        assert "gex" in result.columns

    def test_dte_filter_empties_returns_empty_schema(self):
        """All contracts past dte_max → empty, schema-stable."""
        from shared.analysis_gex import compute_gex_per_strike

        leap_only = _make_gex_df(
            call_strikes=[(100.0, 1000)],
            expiry=date.today() + timedelta(days=365),
        )
        result = compute_gex_per_strike(leap_only, dte_max=45)
        assert len(result) == 0
        assert "strike" in result.columns
