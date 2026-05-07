"""Position-aware OPRA option symbol parser.

Replaces the buggy `str.contains("C")` / `str.contains("P")` pattern that
misclassifies any underlying containing C or P (MCD, KC, CF, PFE, ...).

OPRA standard format: <ROOT><YYMMDD><C|P><STRIKE×1000 zero-padded to 8>
  Example: AAPL250117C00150000  →  AAPL, 2025-01-17, call, $150.00
  Example: MCD260321P00050000   →  MCD,  2026-03-21, put,  $50.00

The C/P indicator sits at a fixed offset from the end of the symbol
(position 9 from the right), not anywhere a C or P happens to appear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Final

import pandas as pd

# ROOT (1-6 upper-case letters) + YYMMDD + C|P + 8-digit strike
_OPRA_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<root>[A-Z]{1,6})(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$"
)


@dataclass(frozen=True)
class OptionSymbolInfo:
    """Parsed OPRA option symbol. Immutable per project coding-style.md."""

    underlying: str
    expiry: date
    option_type: str  # "call" or "put"
    strike: float


# OPRA 2-digit years: convention maps YY < 50 → 20YY, YY >= 50 → 19YY would
# follow ISO; but in practice all UW data is post-2000 and option contracts
# don't list >25y forward. We treat the full 00–99 range as 20YY and reject
# implausibly far years (>30y from now) to surface garbage symbols early.
_MAX_REASONABLE_YEAR_OFFSET: Final[int] = 30


def parse_option_symbol(symbol: str) -> OptionSymbolInfo:
    """Parse an OPRA option symbol into its structural fields.

    The C/P indicator is read from its OPRA-fixed position (right before the
    8-digit strike), not from a substring search — this is what fixes the
    pre-refactor bug where MCD/KC/CF puts were misclassified as calls.

    Raises:
        ValueError: if the symbol is None, empty, malformed, or carries an
            implausibly far expiry year (likely garbage data).
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"Invalid option symbol: {symbol!r}")
    m = _OPRA_RE.match(symbol)
    if m is None:
        raise ValueError(f"Symbol does not match OPRA format: {symbol!r}")
    yy, mm, dd = int(m.group("yy")), int(m.group("mm")), int(m.group("dd"))
    year = 2000 + yy
    today = date.today()
    if year - today.year > _MAX_REASONABLE_YEAR_OFFSET:
        raise ValueError(
            f"Implausible expiry year {year} in symbol {symbol!r} "
            f"(>{_MAX_REASONABLE_YEAR_OFFSET}y from today)"
        )
    try:
        expiry = date(year, mm, dd)
    except ValueError as e:
        raise ValueError(f"Invalid expiry date in symbol {symbol!r}: {e}") from e
    option_type = "call" if m.group("cp") == "C" else "put"
    strike = int(m.group("strike")) / 1000.0
    return OptionSymbolInfo(
        underlying=m.group("root"),
        expiry=expiry,
        option_type=option_type,
        strike=strike,
    )


def _check_series_dtype(symbols: pd.Series) -> None:
    if not (
        pd.api.types.is_string_dtype(symbols)
        or pd.api.types.is_object_dtype(symbols)
    ):
        raise TypeError(
            f"Expected string/object Series, got dtype={symbols.dtype}"
        )


def classify_option_types(symbols: pd.Series) -> pd.Series:
    """Vectorised call/put classification for a Series of OPRA symbols.

    Returns a Series aligned to the input index, mapping each symbol to
    "call" / "put" / NaN. Position-aware: extracts the C/P at the OPRA-
    specified offset rather than searching for any C anywhere in the string.

    Raises:
        TypeError: if the input is not a string/object-dtype Series — guards
            against silently producing all-NaN output from numeric/bool input.
    """
    _check_series_dtype(symbols)
    if len(symbols) == 0:
        return pd.Series([], dtype="object", index=symbols.index)
    extracted = symbols.astype("string").str.extract(_OPRA_RE)
    cp = extracted["cp"]
    return cp.map({"C": "call", "P": "put"})


def extract_option_expiries(symbols: pd.Series) -> pd.Series:
    """Vectorised expiry extraction. Returns ISO date strings (YYYY-MM-DD) or NaN.

    Position-aware (uses the OPRA YYMMDD field, not a substring search).
    Calendar-validates the extracted date — symbols with invalid months/days
    yield NaN rather than malformed strings like "2026-13-45".
    """
    _check_series_dtype(symbols)
    if len(symbols) == 0:
        return pd.Series([], dtype="object", index=symbols.index)
    extracted = symbols.astype("string").str.extract(_OPRA_RE)
    yy, mm, dd = extracted["yy"], extracted["mm"], extracted["dd"]
    iso_raw = "20" + yy + "-" + mm + "-" + dd
    parsed = pd.to_datetime(iso_raw, format="%Y-%m-%d", errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), other=pd.NA)
