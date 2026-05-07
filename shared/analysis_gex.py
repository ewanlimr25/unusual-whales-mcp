"""Gamma Exposure (GEX) primitives.

Extracted from the duplicated inline GEX logic in:
  - servers/options_flow.py:556-625 (today_gamma_flip)
  - servers/options_flow.py:748-819 (gamma_exposure_profile)

Adds optional `dte_max` parameter to address audit §2.2 LEAP-bias issue.
GEX is dominated by near-term gamma (1/sqrt(T) decay); summing across all
DTEs lets LEAPs distort the zero-gamma level.

Convention: dealer perspective.
    Calls → positive GEX (dealer long gamma when public is long calls)
    Puts  → negative GEX (dealer short gamma when public is long puts)
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from shared.option_symbol import extract_option_expiries


def compute_gex_per_strike(
    df: pd.DataFrame,
    dte_max: int | None = 45,
) -> pd.DataFrame:
    """Compute net dealer GEX per strike.

    Args:
        df: Options data. Must include columns: option_type (lowercase
            'call'/'put'), strike, gamma, open_interest, underlying_price.
            If `dte_max` is set, must also include either `option_symbol`
            (for expiry parsing) or `expiry`.
        dte_max: Filter contracts whose expiry is more than this many days
            out. Default 45 (≈ next two monthly cycles). `None` keeps all.

    Returns:
        DataFrame with columns: strike, gex. Sorted ascending by strike.
        Empty schema-stable DataFrame on empty input.

    Raises:
        ValueError: if `dte_max` is set but neither `expiry` nor
            `option_symbol` is in the input columns.
    """
    schema = ["strike", "gex"]
    if df.empty:
        return pd.DataFrame(columns=schema)

    work = df.copy()
    for col in ("gamma", "open_interest", "underlying_price", "strike"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["gamma", "open_interest", "underlying_price", "strike"])
    work = work[(work["gamma"] > 0) & (work["open_interest"] > 0)]
    if work.empty:
        return pd.DataFrame(columns=schema)

    if dte_max is not None:
        if "expiry" not in work.columns:
            if "option_symbol" not in work.columns:
                raise ValueError(
                    "compute_gex_per_strike with dte_max requires either "
                    "'expiry' or 'option_symbol' in the input"
                )
            work["expiry"] = extract_option_expiries(work["option_symbol"])
        expiry_dt = pd.to_datetime(work["expiry"], errors="coerce")
        today = pd.Timestamp(date.today())
        dte = (expiry_dt - today).dt.days
        work = work[(dte >= 0) & (dte <= dte_max)]
        if work.empty:
            return pd.DataFrame(columns=schema)

    # Per-row GEX with sign convention (calls positive, puts negative)
    work["gex"] = work["gamma"] * work["open_interest"] * 100 * work["underlying_price"]
    work.loc[work["option_type"] == "put", "gex"] = -work.loc[work["option_type"] == "put", "gex"]

    grouped = (
        work.groupby("strike", as_index=False)["gex"]
        .sum()
        .sort_values("strike")
        .reset_index(drop=True)
    )
    return grouped[schema]


def compute_zero_gamma_level(strike_gex: pd.DataFrame) -> float | None:
    """Find the FIRST strike (ascending) at which cumulative GEX crosses zero.

    Linearly interpolates between the two strikes bracketing the sign flip.
    When cumulative GEX lands exactly on a strike, that strike is returned.
    Multiple crossings are discarded — only the first is returned.

    Returns None when cumulative GEX never changes sign.
    """
    if len(strike_gex) < 2:
        return None
    work = strike_gex.sort_values("strike").reset_index(drop=True)
    work["cum"] = work["gex"].cumsum()
    for i in range(1, len(work)):
        g1 = float(work.iloc[i - 1]["cum"])
        g2 = float(work.iloc[i]["cum"])
        if g1 * g2 < 0:
            s1 = float(work.iloc[i - 1]["strike"])
            s2 = float(work.iloc[i]["strike"])
            zgl = s1 + (s2 - s1) * (-g1) / (g2 - g1)
            return round(zgl, 2)
        # Exact-zero crossing — cumulative GEX lands precisely on a strike
        if g2 == 0.0 and g1 != 0.0:
            return float(work.iloc[i]["strike"])
    return None


def classify_gex_regime(
    total_gex: float,
    spot: float,
    zero_gamma_level: float | None,
) -> str:
    """Classify the dealer-hedging regime.

    POSITIVE       — spot strictly above ZGL → dealers long gamma → mean reversion
    NEGATIVE       — spot at or below ZGL → dealers short gamma → trend amplification
    FULLY_POSITIVE — no ZGL crossing, total GEX > 0
    FULLY_NEGATIVE — no ZGL crossing, total GEX < 0
    FLAT           — no ZGL crossing, total GEX exactly zero
    """
    if zero_gamma_level is None:
        if total_gex > 0:
            return "FULLY_POSITIVE"
        if total_gex < 0:
            return "FULLY_NEGATIVE"
        return "FLAT"
    return "POSITIVE" if spot > zero_gamma_level else "NEGATIVE"
