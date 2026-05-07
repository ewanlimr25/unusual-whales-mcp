"""Tests for shared.analysis_greeks — closed-form vanna and charm approximations.

These are approximations, NOT exact BSM. Used for relative cross-sectional
ranking in `vanna_charm_exposure`, not absolute pricing. Per audit §3.2.

vanna ≈ -vega × delta / (σ × S)         (sensitivity of delta to vol)
charm ≈ -theta × delta / option_price   (sensitivity of delta to time)
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class TestComputeVanna:
    def test_atm_call_returns_negative_vanna(self):
        """ATM call: delta=0.5, vega>0, σ>0, S>0 → vanna < 0.
        Per Karsan: dealer short ATM calls means rising IV pushes delta up,
        forcing dealer to buy → mechanical bid (vanna squeeze)."""
        from shared.analysis_greeks import compute_vanna

        v = compute_vanna(delta=0.5, vega=0.10, sigma=0.20, spot=100.0)
        assert v < 0

    def test_atm_put_returns_positive_vanna(self):
        from shared.analysis_greeks import compute_vanna

        v = compute_vanna(delta=-0.5, vega=0.10, sigma=0.20, spot=100.0)
        assert v > 0

    def test_zero_sigma_returns_zero(self):
        """σ=0 is a degenerate case — return 0 rather than divide-by-zero."""
        from shared.analysis_greeks import compute_vanna

        v = compute_vanna(delta=0.5, vega=0.10, sigma=0.0, spot=100.0)
        assert v == 0.0

    def test_zero_spot_returns_zero(self):
        from shared.analysis_greeks import compute_vanna

        v = compute_vanna(delta=0.5, vega=0.10, sigma=0.20, spot=0.0)
        assert v == 0.0

    def test_proportional_to_vega(self):
        from shared.analysis_greeks import compute_vanna

        v1 = compute_vanna(delta=0.5, vega=0.10, sigma=0.20, spot=100.0)
        v2 = compute_vanna(delta=0.5, vega=0.20, sigma=0.20, spot=100.0)
        assert v2 == pytest.approx(2 * v1)


class TestComputeCharm:
    def test_call_with_negative_theta_returns_positive_charm(self):
        """Theta is conventionally negative for long options. For an ITM-ish call
        with positive delta, the formula -theta * delta / price yields positive
        charm (delta drifts up as time passes — consistent with delta → 1 at expiry
        for an ITM call)."""
        from shared.analysis_greeks import compute_charm

        c = compute_charm(delta=0.7, theta=-0.05, option_price=2.0)
        assert c > 0

    def test_put_with_negative_theta_returns_negative_charm(self):
        """ITM-ish put: delta negative, theta negative → charm negative
        (delta drifts toward -1, i.e. becomes more negative)."""
        from shared.analysis_greeks import compute_charm

        c = compute_charm(delta=-0.7, theta=-0.05, option_price=2.0)
        assert c < 0

    def test_zero_option_price_returns_zero(self):
        from shared.analysis_greeks import compute_charm

        c = compute_charm(delta=0.5, theta=-0.05, option_price=0.0)
        assert c == 0.0

    def test_proportional_to_delta(self):
        from shared.analysis_greeks import compute_charm

        c1 = compute_charm(delta=0.3, theta=-0.05, option_price=2.0)
        c2 = compute_charm(delta=0.6, theta=-0.05, option_price=2.0)
        assert c2 == pytest.approx(2 * c1)


class TestVectorisedGreeks:
    def test_compute_vanna_vectorised(self):
        from shared.analysis_greeks import compute_vanna_series

        df = pd.DataFrame({
            "delta": [0.5, -0.5],
            "vega": [0.10, 0.10],
            "implied_volatility": [0.20, 0.20],
            "underlying_price": [100.0, 100.0],
        })
        vanna = compute_vanna_series(df)
        assert vanna.iloc[0] < 0  # ATM call
        assert vanna.iloc[1] > 0  # ATM put

    def test_compute_charm_vectorised(self):
        from shared.analysis_greeks import compute_charm_series

        df = pd.DataFrame({
            "delta": [0.7, -0.7],
            "theta": [-0.05, -0.05],
            "price": [2.0, 2.0],
        })
        charm = compute_charm_series(df)
        assert charm.iloc[0] > 0
        assert charm.iloc[1] < 0

    def test_handles_nan_input(self):
        from shared.analysis_greeks import compute_vanna_series

        df = pd.DataFrame({
            "delta": [0.5, math.nan],
            "vega": [0.10, 0.10],
            "implied_volatility": [0.20, 0.20],
            "underlying_price": [100.0, 100.0],
        })
        vanna = compute_vanna_series(df)
        assert vanna.iloc[0] < 0
        # NaN should propagate to NaN output, not crash
        assert pd.isna(vanna.iloc[1])

    def test_vectorised_handles_zero_sigma_per_row(self):
        from shared.analysis_greeks import compute_vanna_series

        df = pd.DataFrame({
            "delta": [0.5, 0.5],
            "vega": [0.10, 0.10],
            "implied_volatility": [0.20, 0.0],  # second row degenerate
            "underlying_price": [100.0, 100.0],
        })
        vanna = compute_vanna_series(df)
        assert vanna.iloc[0] != 0
        assert vanna.iloc[1] == 0.0
