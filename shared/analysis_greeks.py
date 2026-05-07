"""Closed-form approximations for vanna and charm.

These are NOT exact Black-Scholes Greeks — they are first-order approximations
derived from already-stored delta/vega/theta. Used for cross-sectional ranking
in `vanna_charm_exposure`, not absolute pricing or hedging math.

References:
  - Karsan ("Vanna/charm flows") — Forward Guidance / Real Vision interviews
  - SpotGamma "Vanna and Charm" white paper (2021)
  - Demeterfi-Derman-Kamal-Zou 1999 (variance-swap math underpinning)

Conventions:
  vanna = -vega × delta / (σ × S)        sensitivity of delta to vol
  charm = -theta × delta / option_price  sensitivity of delta to time

Sign interpretation (PUBLIC-perspective; dealer hedge is the inverse):
  - Public net negative vanna (call-heavy book): falling IV reduces |delta|,
    dealers (short calls) cut their long-underlying hedge → SELLING pressure.
  - Public net positive vanna (put-heavy book): falling IV reduces |put delta|,
    dealers (short puts) cover by BUYING underlying → vanna-squeeze setup.
  - Charm (∂delta/∂t) is the analogous force from time decay rather than vol;
    it accelerates around OPEX week.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vanna(delta: float, vega: float, sigma: float, spot: float) -> float:
    """Approximate vanna = -vega × delta / (σ × S).

    Returns 0.0 when σ or spot is zero (degenerate inputs).
    """
    if sigma <= 0 or spot <= 0:
        return 0.0
    return -vega * delta / (sigma * spot)


def compute_charm(delta: float, theta: float, option_price: float) -> float:
    """Approximate charm = -theta × delta / option_price.

    Returns 0.0 when option_price is zero (degenerate input).
    """
    if option_price <= 0:
        return 0.0
    return -theta * delta / option_price


def compute_vanna_series(df: pd.DataFrame) -> pd.Series:
    """Vectorised vanna. Required columns: delta, vega, implied_volatility,
    underlying_price.

    Returns a Series aligned to `df.index`. Behaviour at the boundaries:
      - sigma <= 0 or spot <= 0   → 0.0 (degenerate denominator)
      - any input column is NaN   → NaN propagates (callers must dropna first)
    """
    delta = pd.to_numeric(df["delta"], errors="coerce")
    vega = pd.to_numeric(df["vega"], errors="coerce")
    sigma = pd.to_numeric(df["implied_volatility"], errors="coerce")
    spot = pd.to_numeric(df["underlying_price"], errors="coerce")
    denom = sigma * spot
    # avoid divide-by-zero — anywhere denom is non-positive, output 0
    safe = denom.where(denom > 0)
    vanna = -vega * delta / safe
    return vanna.where(denom > 0, other=0.0)


def compute_charm_series(df: pd.DataFrame) -> pd.Series:
    """Vectorised charm. Required columns: delta, theta, price.

    Returns a Series aligned to `df.index`. Boundary behaviour:
      - price <= 0              → 0.0 (degenerate denominator)
      - any input column is NaN → NaN propagates (callers must dropna first)
    """
    delta = pd.to_numeric(df["delta"], errors="coerce")
    theta = pd.to_numeric(df["theta"], errors="coerce")
    price = pd.to_numeric(df["price"], errors="coerce")
    safe = price.where(price > 0)
    charm = -theta * delta / safe
    return charm.where(price > 0, other=0.0)
