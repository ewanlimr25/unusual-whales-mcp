"""Tests for shared.dp_classification — canonical dark pool buy/sell classification.

Replaces 4 duplicated implementations (insights.py:362-376, insights.py:484-504,
risk.py:189-198, strategy.py:75-88) which had subtle threshold drift. The shared
module provides primitives; consumer-specific thresholds (1.3, 1.5, 0.6/0.4)
remain in callers.

Convention: a trade is a BUY when price >= NBBO mid, SELL when price < NBBO mid.
This matches all 4 prior implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# classify_dp_side — adds a side column
# ============================================================================


class TestClassifyDpSide:
    def test_at_mid_classified_as_buy(self):
        """Convention: price == mid is a buy (matches all 4 legacy callers)."""
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame({
            "price": [100.0],
            "nbbo_bid": [99.0],
            "nbbo_ask": [101.0],
            "size": [1000],
        })
        result = classify_dp_side(df)
        assert result.loc[0, "side"] == "buy"

    def test_above_mid_is_buy(self):
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame({
            "price": [100.5],
            "nbbo_bid": [99.0],
            "nbbo_ask": [101.0],
            "size": [1000],
        })
        result = classify_dp_side(df)
        assert result.loc[0, "side"] == "buy"

    def test_below_mid_is_sell(self):
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame({
            "price": [99.5],
            "nbbo_bid": [99.0],
            "nbbo_ask": [101.0],
            "size": [1000],
        })
        result = classify_dp_side(df)
        assert result.loc[0, "side"] == "sell"

    def test_does_not_mutate_input(self):
        """Per coding-style.md: never mutate; return a new DataFrame."""
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame({
            "price": [100.0],
            "nbbo_bid": [99.0],
            "nbbo_ask": [101.0],
            "size": [1000],
        })
        original_cols = list(df.columns)
        _ = classify_dp_side(df)
        assert list(df.columns) == original_cols
        assert "side" not in df.columns

    def test_empty_dataframe(self):
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame(columns=["price", "nbbo_bid", "nbbo_ask", "size"])
        result = classify_dp_side(df)
        assert "side" in result.columns
        assert len(result) == 0

    def test_handles_string_numeric_columns(self):
        """UW Parquet sometimes has numeric columns stored as strings."""
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame({
            "price": ["100.5", "99.5"],
            "nbbo_bid": ["99.0", "99.0"],
            "nbbo_ask": ["101.0", "101.0"],
            "size": ["1000", "500"],
        })
        result = classify_dp_side(df)
        assert result.loc[0, "side"] == "buy"
        assert result.loc[1, "side"] == "sell"

    def test_missing_required_column_raises(self):
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame({"price": [100.0]})
        with pytest.raises((KeyError, ValueError)):
            classify_dp_side(df)

    def test_nan_price_treated_as_sell(self):
        """A row with NaN price/mid should not crash; convention: treat as sell
        (the conservative interpretation) and document."""
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame({
            "price": [np.nan, 100.5],
            "nbbo_bid": [99.0, 99.0],
            "nbbo_ask": [101.0, 101.0],
            "size": [1000, 500],
        })
        result = classify_dp_side(df)
        # NaN row should not be classified as buy
        assert result.loc[0, "side"] != "buy"
        assert result.loc[1, "side"] == "buy"

    def test_nan_mid_treated_as_sell(self):
        """When bid or ask is NaN, mid is NaN → row is conservatively classified as sell."""
        from shared.dp_classification import classify_dp_side

        df = pd.DataFrame({
            "price": [100.5, 100.5],
            "nbbo_bid": [np.nan, 99.0],
            "nbbo_ask": [101.0, np.nan],
            "size": [1000, 500],
        })
        result = classify_dp_side(df)
        assert result.loc[0, "side"] == "sell"
        assert result.loc[1, "side"] == "sell"


# ============================================================================
# aggregate_dp_per_ticker — per-ticker buy/sell aggregation
# ============================================================================


class TestAggregateDpPerTicker:
    def test_basic_aggregation(self):
        from shared.dp_classification import aggregate_dp_per_ticker

        df = pd.DataFrame({
            "ticker": ["AAPL", "AAPL", "AAPL", "TSLA"],
            "price": [100.5, 100.5, 99.5, 200.5],
            "nbbo_bid": [99.0, 99.0, 99.0, 199.0],
            "nbbo_ask": [101.0, 101.0, 101.0, 201.0],
            "size": [1000, 500, 800, 200],
            "premium": [100500, 50250, 79600, 40100],
        })
        result = aggregate_dp_per_ticker(df)

        aapl = result[result["ticker"] == "AAPL"].iloc[0]
        assert aapl["buy_volume"] == 1500  # 1000 + 500
        assert aapl["sell_volume"] == 800
        assert aapl["total_volume"] == 2300
        assert abs(aapl["buy_ratio"] - 1500 / 2300) < 1e-9

    def test_empty_input_returns_empty(self):
        from shared.dp_classification import aggregate_dp_per_ticker

        df = pd.DataFrame(columns=["ticker", "price", "nbbo_bid", "nbbo_ask", "size", "premium"])
        result = aggregate_dp_per_ticker(df)
        assert len(result) == 0
        # Schema must be stable so callers can index columns without checks
        for col in ["ticker", "buy_volume", "sell_volume", "total_volume", "buy_ratio"]:
            assert col in result.columns

    def test_zero_volume_buy_ratio_is_neutral(self):
        """Convention: 0/0 → 0.5 (neutral). Matches insights.py:498 default."""
        from shared.dp_classification import aggregate_dp_per_ticker

        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [100.0],
            "nbbo_bid": [99.0],
            "nbbo_ask": [101.0],
            "size": [0],
            "premium": [0.0],
        })
        result = aggregate_dp_per_ticker(df)
        aapl = result[result["ticker"] == "AAPL"].iloc[0]
        # All-zero size → ratio defaults to neutral
        assert aapl["buy_ratio"] == 0.5

    def test_includes_premium_aggregate(self):
        from shared.dp_classification import aggregate_dp_per_ticker

        df = pd.DataFrame({
            "ticker": ["AAPL", "AAPL"],
            "price": [100.5, 99.5],
            "nbbo_bid": [99.0, 99.0],
            "nbbo_ask": [101.0, 101.0],
            "size": [1000, 800],
            "premium": [100500.0, 79600.0],
        })
        result = aggregate_dp_per_ticker(df)
        aapl = result[result["ticker"] == "AAPL"].iloc[0]
        assert aapl["total_premium"] == pytest.approx(180100.0)

    def test_classification_label_at_exact_threshold(self):
        """Boundary: ratio == 0.60 → BUY (not NEUTRAL); ratio == 0.40 → SELL."""
        from shared.dp_classification import _label_for_ratio

        assert _label_for_ratio(0.60) == "BUY"
        assert _label_for_ratio(0.40) == "SELL"
        assert _label_for_ratio(0.59) == "NEUTRAL"
        assert _label_for_ratio(0.41) == "NEUTRAL"

    def test_classification_label(self):
        """Convenience label: BUY (>=0.6), SELL (<=0.4), NEUTRAL otherwise.
        Matches conviction_matrix's bull_threshold=0.6 / bear_threshold=0.4."""
        from shared.dp_classification import aggregate_dp_per_ticker

        df = pd.DataFrame({
            "ticker": ["BUY1", "SELL1", "MIXED"],
            "price": [100.5, 99.5, 100.0],
            "nbbo_bid": [99.0, 99.0, 99.0],
            "nbbo_ask": [101.0, 101.0, 101.0],
            "size": [1000, 1000, 1000],
            "premium": [100500.0, 79600.0, 80000.0],
        })
        result = aggregate_dp_per_ticker(df)
        labels = dict(zip(result["ticker"], result["classification"]))
        assert labels["BUY1"] == "BUY"
        assert labels["SELL1"] == "SELL"

    def test_does_not_mutate_input(self):
        from shared.dp_classification import aggregate_dp_per_ticker

        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "price": [100.5],
            "nbbo_bid": [99.0],
            "nbbo_ask": [101.0],
            "size": [1000],
            "premium": [100500.0],
        })
        original_cols = list(df.columns)
        _ = aggregate_dp_per_ticker(df)
        assert list(df.columns) == original_cols

    def test_filter_by_ticker(self):
        """Optional filter so callers don't need a separate slice."""
        from shared.dp_classification import aggregate_dp_per_ticker

        df = pd.DataFrame({
            "ticker": ["AAPL", "TSLA"],
            "price": [100.5, 200.5],
            "nbbo_bid": [99.0, 199.0],
            "nbbo_ask": [101.0, 201.0],
            "size": [1000, 500],
            "premium": [100500.0, 100250.0],
        })
        result = aggregate_dp_per_ticker(df, ticker="AAPL")
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"


# ============================================================================
# Single-ticker convenience: compute_buy_ratio
# ============================================================================


class TestComputeBuyRatio:
    def test_returns_float_in_zero_one(self):
        from shared.dp_classification import compute_buy_ratio

        df = pd.DataFrame({
            "price": [100.5, 100.5, 99.5],
            "nbbo_bid": [99.0, 99.0, 99.0],
            "nbbo_ask": [101.0, 101.0, 101.0],
            "size": [1000, 500, 800],
        })
        ratio = compute_buy_ratio(df)
        assert 0.0 <= ratio <= 1.0
        assert ratio == pytest.approx(1500 / 2300)

    def test_empty_returns_neutral(self):
        from shared.dp_classification import compute_buy_ratio

        df = pd.DataFrame(columns=["price", "nbbo_bid", "nbbo_ask", "size"])
        assert compute_buy_ratio(df) == 0.5

    def test_all_buys(self):
        from shared.dp_classification import compute_buy_ratio

        df = pd.DataFrame({
            "price": [100.5, 100.5],
            "nbbo_bid": [99.0, 99.0],
            "nbbo_ask": [101.0, 101.0],
            "size": [1000, 500],
        })
        assert compute_buy_ratio(df) == 1.0

    def test_all_sells(self):
        from shared.dp_classification import compute_buy_ratio

        df = pd.DataFrame({
            "price": [99.5, 99.5],
            "nbbo_bid": [99.0, 99.0],
            "nbbo_ask": [101.0, 101.0],
            "size": [1000, 500],
        })
        assert compute_buy_ratio(df) == 0.0
