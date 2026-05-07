"""Shared pytest fixtures.

Synthetic Parquet fixtures monkeypatch `shared.data_loader.STOCKS_DIR` so
tests can call production `load_data()` against deterministic data without
touching `~/Documents/Stocks/`.

Integration tests (gated by `pytest.mark.integration` and skipped by default)
hit real Parquet files; see `pyproject.toml` `[tool.pytest.ini_options]`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable so tests can `from shared...` and `from servers...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures import synthesise


@pytest.fixture
def stocks_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point shared.data_loader.STOCKS_DIR at a tmp directory."""
    import shared.data_loader as data_loader

    monkeypatch.setattr(data_loader, "STOCKS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def oi_parquet(stocks_dir: Path) -> Path:
    """Write a synthetic OI Parquet to the test stocks dir for date 2026-03-17."""
    df = synthesise.build_oi_df()
    return synthesise.write_oi_parquet(stocks_dir, "2026-03-17", df)


@pytest.fixture
def oi_df() -> pd.DataFrame:
    """Raw synthetic OI DataFrame for tests that don't need Parquet round-trip."""
    return synthesise.build_oi_df()


@pytest.fixture
def darkpool_parquet(stocks_dir: Path) -> Path:
    df = synthesise.build_darkpool_df()
    return synthesise.write_darkpool_parquet(stocks_dir, "2026-03-17", df)


@pytest.fixture
def darkpool_df() -> pd.DataFrame:
    return synthesise.build_darkpool_df()


@pytest.fixture
def screener_history(stocks_dir: Path):
    """Write screener Parquet files for 30 consecutive dates with deterministic
    IV30d / P-C distributions. Returns a callable to fetch the per-date dicts.

    AAPL: rising IV (0.10 to 0.40 over 30 days) — a multi-day uptrend
    TSLA: stable IV ~ 0.50 with one outlier spike to 5.0 (testing percentile
          robustness vs IV-rank)
    SPY:  ascending P/C ratio for z-score sentiment tests
    """
    import numpy as np

    rng = np.random.default_rng(42)
    dates = []
    for i in range(30):
        # Dates in descending order from "today" backwards
        date = f"2026-03-{17 - (29 - i):02d}" if (17 - (29 - i)) > 0 else f"2026-02-{17 - (29 - i) + 28:02d}"
        # Use simple monotonic dates
        date = f"2026-{(2 + i // 31):02d}-{((i % 31) + 1):02d}"
        dates.append(date)

    # IV trajectories
    aapl_iv = np.linspace(0.10, 0.40, 30)
    tsla_iv = np.full(30, 0.50)
    tsla_iv[15] = 5.0  # outlier
    spy_pcr = np.linspace(0.6, 1.4, 30)

    for i, date in enumerate(dates):
        iv_per = {"AAPL": float(aapl_iv[i]), "TSLA": float(tsla_iv[i]), "SPY": 0.20}
        pcr_per = {"AAPL": 1.0, "TSLA": 1.0, "SPY": float(spy_pcr[i])}
        df = synthesise.build_screener_df_for_date(date, iv_per, pcr_per)
        synthesise.write_screener_parquet(stocks_dir, date, df)

    return dates
