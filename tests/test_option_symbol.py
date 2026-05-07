"""Tests for shared.option_symbol — position-aware OPRA parser.

Replaces the buggy `str.contains("C")` pattern in oi_changes.py:165, 196.
The bug misclassifies any underlying with `C` in the symbol (MCD, KC, CF, etc.).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# Single-symbol parser
# ============================================================================


class TestParseOptionSymbol:
    def test_standard_call(self):
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("AAPL250117C00150000")
        assert info.underlying == "AAPL"
        assert info.expiry == date(2025, 1, 17)
        assert info.option_type == "call"
        assert info.strike == 150.0

    def test_standard_put(self):
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("SPY250117P00450000")
        assert info.underlying == "SPY"
        assert info.option_type == "put"
        assert info.strike == 450.0

    def test_mcd_call_not_misclassified(self):
        """MCD has a C in the underlying. The old bug classified this and any other
        symbol containing C as a call regardless of position. Verify the parser
        looks at the C/P indicator in its proper OPRA position."""
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("MCD260321C00050000")
        assert info.underlying == "MCD"
        assert info.option_type == "call"

    def test_mcd_put_not_misclassified_as_call(self):
        """The original bug: MCD put was classified as a call because 'C' is in 'MCD'."""
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("MCD260321P00050000")
        assert info.underlying == "MCD"
        assert info.option_type == "put"
        assert info.strike == 50.0

    def test_kc_two_char_root_put(self):
        """KC (coffee) — 2-char root, both letters non-K matter. Old bug: 'K' or 'C'
        in the symbol would falsely classify."""
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("KC260321P00150000")
        assert info.underlying == "KC"
        assert info.option_type == "put"

    def test_cf_root_call(self):
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("CF260321C00050000")
        assert info.underlying == "CF"
        assert info.option_type == "call"

    def test_long_root_googl(self):
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("GOOGL260321C02000000")
        assert info.underlying == "GOOGL"
        assert info.strike == 2000.0
        assert info.option_type == "call"

    def test_strike_with_decimals(self):
        """Strike encoding: 8 digits, last 3 are thousandths."""
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("AAPL250117C00150500")
        assert info.strike == 150.500

    def test_strike_fractional(self):
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("AAPL250117C00037500")
        assert info.strike == 37.500

    def test_real_world_vix_sample(self):
        """Sample taken from production data."""
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("VIX260519C00070000")
        assert info.underlying == "VIX"
        assert info.expiry == date(2026, 5, 19)
        assert info.option_type == "call"
        assert info.strike == 70.0

    def test_real_world_iwm_put(self):
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("IWM260320P00240000")
        assert info.underlying == "IWM"
        assert info.option_type == "put"
        assert info.strike == 240.0

    def test_malformed_raises(self):
        from shared.option_symbol import parse_option_symbol

        with pytest.raises(ValueError):
            parse_option_symbol("not_a_symbol")

    def test_empty_raises(self):
        from shared.option_symbol import parse_option_symbol

        with pytest.raises(ValueError):
            parse_option_symbol("")

    def test_none_raises(self):
        from shared.option_symbol import parse_option_symbol

        with pytest.raises(ValueError):
            parse_option_symbol(None)  # type: ignore[arg-type]

    def test_missing_strike_raises(self):
        from shared.option_symbol import parse_option_symbol

        with pytest.raises(ValueError):
            parse_option_symbol("AAPL250117C")

    def test_lowercase_raises_or_normalises(self):
        """Document behaviour: OPRA is upper-case. We reject lower-case rather than guess."""
        from shared.option_symbol import parse_option_symbol

        with pytest.raises(ValueError):
            parse_option_symbol("aapl250117c00150000")

    def test_immutable_dataclass(self):
        """Frozen dataclass — immutability per project coding-style.md."""
        from shared.option_symbol import parse_option_symbol

        info = parse_option_symbol("AAPL250117C00150000")
        with pytest.raises(AttributeError):
            info.underlying = "NEW"  # type: ignore[misc]

    def test_implausibly_far_year_raises(self):
        """A YY that resolves to >30y in the future is treated as garbage data."""
        from shared.option_symbol import parse_option_symbol

        # YY=99 → 2099, well past +30y from 2026
        with pytest.raises(ValueError, match="Implausible"):
            parse_option_symbol("AAPL990117C00150000")

    def test_invalid_calendar_date_raises(self):
        """Month=13 / day=32 etc. must raise rather than silently produce garbage."""
        from shared.option_symbol import parse_option_symbol

        with pytest.raises(ValueError):
            parse_option_symbol("AAPL251332C00150000")


# ============================================================================
# Vectorised helper
# ============================================================================


class TestClassifyOptionTypes:
    def test_basic_classification(self):
        from shared.option_symbol import classify_option_types

        series = pd.Series(
            ["AAPL250117C00150000", "SPY250117P00450000", "MCD260321P00050000"]
        )
        result = classify_option_types(series)
        assert result.tolist() == ["call", "put", "put"]

    def test_consistency_with_single_parser(self):
        """Vectorised helper must agree with parse_option_symbol on every symbol."""
        from shared.option_symbol import classify_option_types, parse_option_symbol

        symbols = [
            "AAPL250117C00150000",
            "SPY250117P00450000",
            "MCD260321C00050000",
            "MCD260321P00050000",
            "KC260321P00150000",
            "CF260321C00050000",
            "GOOGL260321C02000000",
        ]
        series = pd.Series(symbols)
        vec = classify_option_types(series)
        single = [parse_option_symbol(s).option_type for s in symbols]
        assert vec.tolist() == single

    def test_handles_nan(self):
        from shared.option_symbol import classify_option_types

        series = pd.Series(["AAPL250117C00150000", None, "SPY250117P00450000"])
        result = classify_option_types(series)
        # NaN/None must produce a sentinel (None or NaN), not crash
        assert result.iloc[0] == "call"
        assert pd.isna(result.iloc[1]) or result.iloc[1] is None
        assert result.iloc[2] == "put"

    def test_handles_malformed_without_crash(self):
        from shared.option_symbol import classify_option_types

        series = pd.Series(["AAPL250117C00150000", "garbage", "SPY250117P00450000"])
        result = classify_option_types(series)
        assert result.iloc[0] == "call"
        assert pd.isna(result.iloc[1]) or result.iloc[1] is None
        assert result.iloc[2] == "put"

    def test_empty_series(self):
        from shared.option_symbol import classify_option_types

        result = classify_option_types(pd.Series([], dtype=str))
        assert len(result) == 0

    def test_mcd_regression_at_scale(self):
        """The original bug: every MCD put across a full-market dataset was misclassified.
        At scale this manifests as a systemic put → call mislabel."""
        from shared.option_symbol import classify_option_types

        # 100 MCD puts — every one must be "put"
        series = pd.Series(["MCD260321P00050000"] * 100)
        result = classify_option_types(series)
        assert (result == "put").all()

    def test_rejects_numeric_dtype(self):
        """Numeric Series produces silent all-NaN otherwise; reject at the boundary."""
        from shared.option_symbol import classify_option_types

        series = pd.Series([1, 2, 3])
        with pytest.raises(TypeError, match="string/object"):
            classify_option_types(series)

    def test_extract_option_expiries_basic(self):
        from shared.option_symbol import extract_option_expiries

        series = pd.Series([
            "AAPL250117C00150000",
            "MCD260321P00050000",
            "garbage",
        ])
        result = extract_option_expiries(series)
        assert result.iloc[0] == "2025-01-17"
        assert result.iloc[1] == "2026-03-21"
        assert pd.isna(result.iloc[2])

    def test_extract_option_expiries_empty(self):
        from shared.option_symbol import extract_option_expiries

        result = extract_option_expiries(pd.Series([], dtype=str))
        assert len(result) == 0

    def test_preserves_caller_index(self):
        """Result aligns to the caller's index, not a default RangeIndex.
        Regression for the empty-Series case which previously dropped the index."""
        from shared.option_symbol import classify_option_types

        # Non-empty
        series = pd.Series(
            ["AAPL250117C00150000", "SPY250117P00450000"],
            index=[100, 200],
        )
        result = classify_option_types(series)
        assert list(result.index) == [100, 200]

        # Empty
        empty = pd.Series([], dtype=str, index=pd.Index([], dtype="int64"))
        result_empty = classify_option_types(empty)
        assert list(result_empty.index) == []
