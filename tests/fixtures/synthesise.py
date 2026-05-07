"""Synthetic Parquet builders for tests.

Generates small, deterministic Parquet files with the same column names and
dtypes as the production UW exports. Tests monkeypatch
`shared.data_loader.STOCKS_DIR` to point at a tmp directory containing these.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Production folder names (matches shared.data_loader.FOLDER_MAP)
FOLDER_OPTIONS = "All Options"
FOLDER_DARKPOOL = "Dark pool"
FOLDER_HOTCHAINS = "Hot Option Chains"
FOLDER_SCREENER = "Stock Screener"
FOLDER_OI = "OI changes"


def _write_parquet_via_duckdb(df: pd.DataFrame, path: Path) -> None:
    """Use DuckDB to write Parquet — avoids the pyarrow/fastparquet dependency
    that pandas.DataFrame.to_parquet requires. DuckDB is already a project dep."""
    con = duckdb.connect()
    con.register("tmp_df", df)
    con.execute(f"COPY tmp_df TO '{path}' (FORMAT PARQUET)")
    con.close()


def write_oi_parquet(stocks_dir: Path, date: str, df: pd.DataFrame) -> Path:
    folder = stocks_dir / FOLDER_OI
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"chain-oi-changes-{date}.parquet"
    _write_parquet_via_duckdb(df, path)
    return path


def write_darkpool_parquet(stocks_dir: Path, date: str, df: pd.DataFrame) -> Path:
    folder = stocks_dir / FOLDER_DARKPOOL
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"dp-eod-report-{date}.parquet"
    _write_parquet_via_duckdb(df, path)
    return path


def write_options_parquet(stocks_dir: Path, date: str, df: pd.DataFrame) -> Path:
    folder = stocks_dir / FOLDER_OPTIONS
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"bot-eod-report-{date}.parquet"
    _write_parquet_via_duckdb(df, path)
    return path


def write_screener_parquet(stocks_dir: Path, date: str, df: pd.DataFrame) -> Path:
    folder = stocks_dir / FOLDER_SCREENER
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"stock-screener-{date}.parquet"
    _write_parquet_via_duckdb(df, path)
    return path


# ============================================================================
# Synthetic data builders
# ============================================================================


def build_oi_df(seed: int = 42) -> pd.DataFrame:
    """Build a deterministic OI DataFrame.

    Includes:
      - MCD puts (regression for the OPRA `str.contains("C")` bug)
      - Mix of near-DTE (<= 14) and far-DTE (> 30) for OPEX-concentration tests
      - One synthetic position-roll case (near OI down, far OI up on same ticker)
      - 5 tickers total: AAPL, MCD, KC, GOOGL, SPY
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # AAPL — heavy near-OPEX call build (concentration test).
    # Strike 160 dominates (closest to spot 162) so pin-distance tests have
    # a deterministic max-OI strike near spot.
    for strike in [150.0, 155.0, 160.0, 165.0]:
        for opt_type, cp in [("call", "C"), ("put", "P")]:
            yymmdd = "260320"  # near-OPEX (3rd Friday of March 2026)
            sym = f"AAPL{yymmdd}{cp}{int(strike * 1000):08d}"
            # Higher OI at 160 to break ties for pin-distance tests
            base_oi = 12000 if strike == 160.0 else 7000
            rows.append({
                "option_symbol": sym,
                "underlying_symbol": "AAPL",
                "strike": strike,
                "last_oi": 5000,
                "curr_oi": base_oi if opt_type == "call" else int(base_oi * 0.6),
                "oi_diff_plain": 2000.0 if opt_type == "call" else -1000.0,
                "volume": 1500,
                "oi_change": 0.4,
                "avg_price": 5.0,
                "prev_total_premium": 250000.0,
                "prev_ask_volume": 800.0 if opt_type == "call" else 200.0,
                "prev_bid_volume": 200.0 if opt_type == "call" else 800.0,
                "sector": "Technology",
                "stock_price": 162.0,
                "dte": 14,
            })

    # MCD — REGRESSION: must classify these puts as PUTS, not CALLS.
    # Old `str.contains("C")` bug labeled them all as calls because of the C in MCD.
    for strike in [50.0, 52.5, 55.0]:
        sym = f"MCD260320P{int(strike * 1000):08d}"
        rows.append({
            "option_symbol": sym,
            "underlying_symbol": "MCD",
            "strike": strike,
            "last_oi": 1000,
            "curr_oi": 2500,
            "oi_diff_plain": 1500.0,
            "volume": 800,
            "oi_change": 1.5,
            "avg_price": 1.2,
            "prev_total_premium": 96000.0,
            "prev_ask_volume": 600.0,  # ask-heavy: bearish for puts
            "prev_bid_volume": 100.0,
            "sector": "Consumer",
            "stock_price": 50.5,
            "dte": 14,
        })

    # KC — both call and put, two-letter root with K and C in it
    for opt_type, cp, strike in [("call", "C", 150.0), ("put", "P", 145.0)]:
        sym = f"KC260320{cp}{int(strike * 1000):08d}"
        rows.append({
            "option_symbol": sym,
            "underlying_symbol": "KC",
            "strike": strike,
            "last_oi": 500,
            "curr_oi": 800,
            "oi_diff_plain": 300.0,
            "volume": 200,
            "oi_change": 0.6,
            "avg_price": 0.8,
            "prev_total_premium": 24000.0,
            "prev_ask_volume": 150.0,
            "prev_bid_volume": 50.0,
            "sector": "Consumer",
            "stock_price": 147.0,
            "dte": 14,
        })

    # GOOGL — synthetic position-roll case: near calls down, far calls up
    rows.append({
        "option_symbol": "GOOGL260320C00200000",
        "underlying_symbol": "GOOGL",
        "strike": 200.0,
        "last_oi": 5000,
        "curr_oi": 1000,
        "oi_diff_plain": -4000.0,  # near OI DOWN
        "volume": 4500,
        "oi_change": -0.8,
        "avg_price": 3.0,
        "prev_total_premium": 1500000.0,
        "prev_ask_volume": 100.0,
        "prev_bid_volume": 4000.0,
        "sector": "Technology",
        "stock_price": 205.0,
        "dte": 14,  # near
    })
    rows.append({
        "option_symbol": "GOOGL260619C00210000",
        "underlying_symbol": "GOOGL",
        "strike": 210.0,
        "last_oi": 500,
        "curr_oi": 4500,
        "oi_diff_plain": 4000.0,  # far OI UP
        "volume": 4200,
        "oi_change": 8.0,
        "avg_price": 6.0,
        "prev_total_premium": 2500000.0,
        "prev_ask_volume": 3800.0,
        "prev_bid_volume": 200.0,
        "sector": "Technology",
        "stock_price": 205.0,
        "dte": 90,  # far
    })

    # SPY — far-DTE LEAP, low concentration vs near
    for strike in [450.0, 460.0]:
        sym = f"SPY270115C{int(strike * 1000):08d}"
        rows.append({
            "option_symbol": sym,
            "underlying_symbol": "SPY",
            "strike": strike,
            "last_oi": 100,
            "curr_oi": 200,
            "oi_diff_plain": 100.0,
            "volume": 50,
            "oi_change": 1.0,
            "avg_price": 25.0,
            "prev_total_premium": 50000.0,
            "prev_ask_volume": 30.0,
            "prev_bid_volume": 20.0,
            "sector": "Index",
            "stock_price": 455.0,
            "dte": 365,  # LEAP
        })

    df = pd.DataFrame(rows)
    # Ensure deterministic order
    return df.sort_values(["underlying_symbol", "option_symbol"]).reset_index(drop=True)


def build_screener_df_for_date(
    date: str,
    iv30d_per_ticker: dict[str, float] | None = None,
    pcr_per_ticker: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a screener DataFrame for a single date.

    Lets multi-date tests construct a deterministic IV / P-C history per ticker.
    """
    iv30d_per_ticker = iv30d_per_ticker or {"AAPL": 0.25, "TSLA": 0.40, "SPY": 0.18}
    pcr_per_ticker = pcr_per_ticker or {t: 1.0 for t in iv30d_per_ticker}
    rows = []
    for ticker, iv in iv30d_per_ticker.items():
        rows.append({
            "ticker": ticker,
            "date": date,
            "full_name": ticker,
            "sector": "Technology",
            "close": 100.0,
            "marketcap": 1_000_000_000,
            "bullish_premium": 100_000.0,
            "bearish_premium": 50_000.0,
            "call_volume": 1000,
            "put_volume": 500,
            "call_premium": 100_000.0,
            "put_premium": 50_000.0,
            "put_call_ratio": pcr_per_ticker.get(ticker, 1.0),
            "iv_rank": 50.0,
            "iv30d": iv,
            "iv30d_1w": iv,
            "iv30d_1m": iv,
            "volatility": iv,
            "implied_move": 5.0,
            "implied_move_perc": 0.05,
            "next_earnings_date": None,
            "er_time": None,
            "avg_30_day_call_volume": 800,
            "avg_30_day_put_volume": 400,
            "total_open_interest": 50000,
        })
    return pd.DataFrame(rows)


def build_darkpool_df(seed: int = 42) -> pd.DataFrame:
    """Build a deterministic dark pool DataFrame spanning all 4 block tiers per ticker.

    Tiers: mega (>$10M), block (>$1M), large (>$100k), retail (<$100k).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # AAPL — buying pressure across all tiers (price >= mid)
    aapl_trades = [
        (12_000_000.0, 100.5, "mega"),     # mega buy
        (5_000_000.0, 100.5, "block"),     # block buy
        (2_000_000.0, 100.5, "block"),     # block buy
        (500_000.0, 100.5, "large"),       # large buy
        (50_000.0, 100.5, "retail"),       # retail buy
        (50_000.0, 99.5, "retail"),        # retail sell (mixed at retail level)
    ]
    for premium, price, _tier in aapl_trades:
        size = int(premium / price)
        rows.append({
            "ticker": "AAPL",
            "executed_at": "2026-03-17 10:00:00",
            "price": price,
            "size": size,
            "premium": premium,
            "nbbo_bid": 99.0,
            "nbbo_ask": 101.0,
            "ext_hour_sold_codes": "",
        })

    # TSLA — selling pressure
    tsla_trades = [
        (8_000_000.0, 199.5, "block"),      # block sell
        (1_500_000.0, 199.5, "block"),      # block sell
        (300_000.0, 199.5, "large"),        # large sell
        (75_000.0, 200.5, "retail"),        # retail buy (counter-flow)
    ]
    for premium, price, _tier in tsla_trades:
        size = int(premium / price)
        rows.append({
            "ticker": "TSLA",
            "executed_at": "2026-03-17 11:00:00",
            "price": price,
            "size": size,
            "premium": premium,
            "nbbo_bid": 199.0,
            "nbbo_ask": 201.0,
            "ext_hour_sold_codes": "",
        })

    # MCD — only retail-tier activity (no institutional signal)
    for _ in range(5):
        rows.append({
            "ticker": "MCD",
            "executed_at": "2026-03-17 12:00:00",
            "price": 50.5,
            "size": 100,
            "premium": 5050.0,
            "nbbo_bid": 50.0,
            "nbbo_ask": 51.0,
            "ext_hour_sold_codes": "",
        })

    return pd.DataFrame(rows)
