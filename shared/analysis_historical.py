"""Multi-date primitives for historical analytics tools.

Used by Phase 3 tools (iv_percentile_zscore, gex_time_series,
volatility_risk_premium) and the signal_backtest methodology fix.

Centralises:
  - iter_dates(data_type, n)           — list available dates, truncated
  - percentile_rank(value, history)    — 0-100 percentile of value in history
  - zscore(value, history)             — z-score (sample std, ddof=1)
  - realised_volatility(close_series)  — annualised σ from close-to-close returns
"""

from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd

from shared.data_loader import list_available_dates  # re-exported for monkeypatching in tests

_TRADING_DAYS_PER_YEAR = 252


def iter_dates(data_type: str, n: int | None = None) -> list[str]:
    """Return the most recent N available dates for a data type, descending.

    Wraps `shared.data_loader.list_available_dates` so callers — and the
    `gex_time_series` cache — have a single iteration entry point.
    """
    dates = list_available_dates(data_type)
    if n is None:
        return list(dates)
    return list(dates[:max(0, int(n))])


def percentile_rank(value: float, history: list[float] | pd.Series | None) -> float | None:
    """Return 0-100 percentile of `value` within `history`.

    Outlier-robust (a single huge value does not collapse all other ranks to
    near-zero, unlike IV-rank's high/low normalisation).

    Returns:
        0–100 percentile when history has at least one finite value.
        None when history is empty / all-NaN.
        50.0 when all history values are identical.
    """
    if history is None:
        return None
    cleaned = [float(x) for x in history if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not cleaned:
        return None
    if all(x == cleaned[0] for x in cleaned):
        return 50.0
    arr = np.array(cleaned, dtype=float)
    # rank = fraction of history strictly below value + half of equal values
    below = float((arr < value).sum())
    equal = float((arr == value).sum())
    rank = (below + 0.5 * equal) / len(arr) * 100.0
    return max(0.0, min(100.0, rank))


def zscore(value: float, history: list[float] | pd.Series | None) -> float | None:
    """Return (value − mean(history)) / std(history) using sample stdev (ddof=1).

    Returns:
        z-score when history has ≥2 finite values and stdev > 0.
        0.0 when stdev is zero.
        None when history is empty.
    """
    if history is None:
        return None
    cleaned = [float(x) for x in history if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not cleaned:
        return None
    mean = statistics.mean(cleaned)
    if len(cleaned) < 2:
        return 0.0
    std = statistics.stdev(cleaned)
    if std == 0:
        return 0.0
    return (float(value) - mean) / std


def realised_volatility(
    close: pd.Series,
    annualisation_factor: int = _TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Annualised realised σ from a series of closing prices.

    Returns None when the input has fewer than 2 prices (no returns derivable).
    """
    if close is None or len(close) < 2:
        return None
    returns = pd.to_numeric(close, errors="coerce").pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.std(ddof=1) * math.sqrt(annualisation_factor))
