"""Canonical dark pool buy/sell classification.

Replaces 4 duplicated implementations with subtle threshold drift:
  - insights.py:362-376 (institutional_accumulation_detector)
  - insights.py:484-504 (conviction_matrix)
  - risk.py:189-198     (signal_confluence)
  - strategy.py:75-88   (_analyze_signals)

Convention (consistent across all 4 legacy callers):
    BUY  iff price >= NBBO mid
    SELL iff price <  NBBO mid

Consumer-specific thresholds (e.g., 1.3× / 1.5× for "accumulation",
0.6 / 0.4 for conviction matrix) remain in callers — this module only
provides primitives.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

# Conviction-label thresholds. Derived from conviction_matrix defaults
# (insights.py:138-145). Callers can override by post-processing buy_ratio.
_BULL_THRESHOLD: Final[float] = 0.60
_BEAR_THRESHOLD: Final[float] = 0.40

_REQUIRED_COLS: Final[tuple[str, ...]] = ("price", "nbbo_bid", "nbbo_ask", "size")


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _coerce_numeric(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Return a copy with the given columns coerced to numeric. NaN on failure."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def classify_dp_side(df: pd.DataFrame) -> pd.DataFrame:
    """Return a NEW DataFrame with an added `side` column ("buy" / "sell").

    Convention: price >= mid → "buy", price < mid → "sell".
    Rows with NaN price or mid are classified as "sell" (conservative).

    Does not mutate the input.
    """
    _validate_columns(df)
    out = _coerce_numeric(df, ("price", "nbbo_bid", "nbbo_ask"))
    mid = (out["nbbo_bid"] + out["nbbo_ask"]) / 2.0
    is_buy = out["price"].ge(mid).fillna(False)
    out["side"] = pd.Series("sell", index=out.index)
    out.loc[is_buy, "side"] = "buy"
    return out


def compute_buy_ratio(df: pd.DataFrame) -> float:
    """Buy-volume / total-volume for a single-ticker DP slice.

    Returns 0.5 (neutral) when total volume is zero.
    """
    if len(df) == 0:
        return 0.5
    classified = classify_dp_side(df)
    classified["size"] = pd.to_numeric(classified["size"], errors="coerce").fillna(0)
    buy_vol = float(classified.loc[classified["side"] == "buy", "size"].sum())
    sell_vol = float(classified.loc[classified["side"] == "sell", "size"].sum())
    total = buy_vol + sell_vol
    if total <= 0:
        return 0.5
    return buy_vol / total


def aggregate_dp_per_ticker(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Per-ticker DP aggregation: buy_volume, sell_volume, total_volume,
    buy_ratio, total_premium, classification.

    The output schema is stable on empty input so callers can index columns
    without conditional logic.

    Args:
        df: Dark pool trades. Must include columns: ticker, price, nbbo_bid,
            nbbo_ask, size, premium.
        ticker: optional symbol filter.

    Does not mutate the input.
    """
    schema = [
        "ticker",
        "buy_volume",
        "sell_volume",
        "total_volume",
        "buy_ratio",
        "total_premium",
        "trade_count",
        "classification",
    ]
    if "ticker" not in df.columns or "premium" not in df.columns:
        raise ValueError("DataFrame must include 'ticker' and 'premium' columns")
    _validate_columns(df)

    if ticker is not None:
        df = df[df["ticker"] == ticker].copy()

    if len(df) == 0:
        return pd.DataFrame(columns=schema)

    classified = classify_dp_side(df)
    classified["size"] = pd.to_numeric(classified["size"], errors="coerce").fillna(0)
    classified["premium"] = pd.to_numeric(classified["premium"], errors="coerce").fillna(0.0)
    classified["_buy_size"] = classified["size"].where(classified["side"] == "buy", 0.0)
    classified["_sell_size"] = classified["size"].where(classified["side"] == "sell", 0.0)

    grouped = classified.groupby("ticker", as_index=False).agg(
        buy_volume=("_buy_size", "sum"),
        sell_volume=("_sell_size", "sum"),
        total_premium=("premium", "sum"),
        trade_count=("size", "count"),
    )
    grouped["total_volume"] = grouped["buy_volume"] + grouped["sell_volume"]
    grouped["buy_ratio"] = grouped.apply(
        lambda r: 0.5 if r["total_volume"] <= 0 else r["buy_volume"] / r["total_volume"],
        axis=1,
    )
    grouped["classification"] = grouped["buy_ratio"].apply(_label_for_ratio)
    return grouped[schema].reset_index(drop=True)


def _label_for_ratio(ratio: float) -> str:
    """Internal label assignment.

    Convention is `>=` on both bounds so the threshold values themselves
    fall on the side of the conviction label, matching the `>=` convention
    used in `classify_dp_side`. NEUTRAL is the open interval (0.40, 0.60).
    """
    if ratio >= _BULL_THRESHOLD:
        return "BUY"
    if ratio <= _BEAR_THRESHOLD:
        return "SELL"
    return "NEUTRAL"
