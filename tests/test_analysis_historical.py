"""Tests for shared.analysis_historical — multi-date primitives.

Used by Phase 3 tools: iv_percentile_zscore, gex_time_series,
volatility_risk_premium, and the signal_backtest methodology fix.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class TestPercentileRank:
    def test_value_at_median_returns_50(self):
        from shared.analysis_historical import percentile_rank

        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile_rank(3.0, history) == pytest.approx(50.0, abs=1.0)

    def test_value_above_all_returns_100(self):
        from shared.analysis_historical import percentile_rank

        assert percentile_rank(10.0, [1.0, 2.0, 3.0]) == pytest.approx(100.0)

    def test_value_below_all_returns_0(self):
        from shared.analysis_historical import percentile_rank

        assert percentile_rank(0.0, [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_outlier_robustness(self):
        """The whole point of percentile (vs IV-rank): one extreme outlier
        does not dominate. A value at the typical mid should remain mid-rank."""
        from shared.analysis_historical import percentile_rank

        # 99 normal values + 1 huge outlier
        history = [0.20] * 99 + [100.0]
        # A typical 0.20 reading should not get demoted to ~0%
        rank = percentile_rank(0.20, history)
        assert rank > 40.0  # well above the IV-rank equivalent (which would be ~0)

    def test_empty_history_returns_none(self):
        from shared.analysis_historical import percentile_rank

        assert percentile_rank(1.0, []) is None

    def test_all_equal_history_returns_50(self):
        """Degenerate case: all values identical. Convention: 50% (neutral)."""
        from shared.analysis_historical import percentile_rank

        assert percentile_rank(5.0, [5.0, 5.0, 5.0]) == pytest.approx(50.0)

    def test_handles_nan_in_history(self):
        from shared.analysis_historical import percentile_rank

        history = [1.0, 2.0, math.nan, 3.0, 4.0]
        rank = percentile_rank(2.5, history)
        assert rank is not None
        assert 0.0 <= rank <= 100.0


class TestZscore:
    def test_value_at_mean_returns_zero(self):
        from shared.analysis_historical import zscore

        assert zscore(2.0, [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_value_one_sigma_above(self):
        from shared.analysis_historical import zscore

        # mean=3, std=sqrt(2) ≈ 1.414 (sample std with ddof=1 over [1,2,3,4,5])
        z = zscore(4.0 + math.sqrt(2.5), [1.0, 2.0, 3.0, 4.0, 5.0])
        # value above mean by ~sqrt(2.5)
        assert z > 0.0

    def test_zero_std_returns_zero(self):
        from shared.analysis_historical import zscore

        assert zscore(5.0, [5.0, 5.0, 5.0]) == 0.0

    def test_empty_history_returns_none(self):
        from shared.analysis_historical import zscore

        assert zscore(1.0, []) is None

    def test_single_element_history_returns_zero(self):
        """One historical reading → stdev undefined. Convention: 0.0 (cannot
        derive deviation), matching the all-equal degenerate case."""
        from shared.analysis_historical import zscore

        assert zscore(5.0, [3.0]) == 0.0


class TestRealisedVolatility:
    def test_typical_30d_annualized(self):
        """Annualized realized vol from close prices. Reference: a price
        series with daily ~1% returns should produce realized ≈ 0.01 * sqrt(252) ≈ 0.16."""
        from shared.analysis_historical import realised_volatility

        rng = np.random.default_rng(42)
        prices = [100.0]
        for _ in range(60):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.01)))
        rv = realised_volatility(pd.Series(prices))
        # Should be in the 0.10–0.25 ballpark for 1% daily returns
        assert 0.05 < rv < 0.30

    def test_zero_volatility_constant_price(self):
        from shared.analysis_historical import realised_volatility

        prices = pd.Series([100.0] * 30)
        assert realised_volatility(prices) == pytest.approx(0.0, abs=1e-9)

    def test_empty_returns_none(self):
        from shared.analysis_historical import realised_volatility

        assert realised_volatility(pd.Series([], dtype=float)) is None

    def test_too_few_prices_returns_none(self):
        from shared.analysis_historical import realised_volatility

        # Only 1 price → can't compute returns
        assert realised_volatility(pd.Series([100.0])) is None


class TestIterDates:
    def test_returns_first_n_dates_in_descending_order(self, monkeypatch):
        from shared import analysis_historical

        def fake_list(_dt):
            return ["2026-03-17", "2026-03-16", "2026-03-13", "2026-03-12"]

        monkeypatch.setattr(analysis_historical, "list_available_dates", fake_list)
        assert analysis_historical.iter_dates("screener", n=2) == ["2026-03-17", "2026-03-16"]

    def test_n_none_returns_all(self, monkeypatch):
        from shared import analysis_historical

        all_dates = ["2026-03-17", "2026-03-16", "2026-03-13"]
        monkeypatch.setattr(analysis_historical, "list_available_dates", lambda _dt: all_dates)
        assert analysis_historical.iter_dates("screener", n=None) == all_dates

    def test_no_dates_returns_empty(self, monkeypatch):
        from shared import analysis_historical

        monkeypatch.setattr(analysis_historical, "list_available_dates", lambda _dt: [])
        assert analysis_historical.iter_dates("screener", n=10) == []
