"""Tests for convert.py CSV → Parquet conversion."""

from pathlib import Path

import duckdb
import pytest

import convert

HEADER = "executed_at,report_flags,canceled,equity_type\n"
PLAIN_ROW = "2026-09-22 13:30:00+00,{},f,ETF\n"
# The CSV sniffer samples only the first 20,480 rows; push the quoted row past it.
SNIFFER_SAMPLE_ROWS = 20_480


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text(HEADER + "".join(rows))


@pytest.fixture
def stocks_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(convert, "STOCKS_DIR", tmp_path)
    monkeypatch.setattr(convert, "FOLDER_MAP", {"options": "All Options"})
    (tmp_path / "All Options").mkdir()
    return tmp_path / "All Options"


@pytest.mark.unit
def test_quoted_multi_flag_row_beyond_sniffer_sample_converts(stocks_dir: Path) -> None:
    # Arrange
    csv_path = stocks_dir / "bot-eod-report-2026-09-22.csv"
    quoted = '2026-09-22 20:18:54+00,"{extended_hours,intermarket_sweep}",f,Index\n'
    _write_csv(csv_path, [PLAIN_ROW] * (SNIFFER_SAMPLE_ROWS + 10) + [quoted])

    # Act
    convert.convert_to_parquet()

    # Assert
    parquet_path = csv_path.with_suffix(".parquet")
    assert not csv_path.exists()
    row = duckdb.sql(
        f"SELECT report_flags, canceled FROM read_parquet('{parquet_path}') "
        "WHERE report_flags LIKE '%,%'"
    ).fetchone()
    assert row == ("{extended_hours,intermarket_sweep}", False)


@pytest.mark.unit
def test_failed_conversion_removes_partial_parquet_and_keeps_csv(
    stocks_dir: Path,
) -> None:
    # Arrange: unquoted embedded comma gives a row with too many columns
    bad_csv = stocks_dir / "bot-eod-report-2026-09-22.csv"
    broken = "2026-09-22 20:18:54+00,{a,b},f,Index\n"
    _write_csv(bad_csv, [PLAIN_ROW] * (SNIFFER_SAMPLE_ROWS + 10) + [broken])
    good_csv = stocks_dir / "bot-eod-report-2026-09-23.csv"
    _write_csv(good_csv, [PLAIN_ROW])

    # Act
    convert.convert_to_parquet()

    # Assert: bad file left untouched for retry, good file still converted
    assert bad_csv.exists()
    assert not bad_csv.with_suffix(".parquet").exists()
    assert good_csv.with_suffix(".parquet").exists()
    assert not good_csv.exists()


@pytest.mark.unit
def test_existing_parquet_is_skipped(stocks_dir: Path) -> None:
    # Arrange
    csv_path = stocks_dir / "bot-eod-report-2026-09-22.csv"
    _write_csv(csv_path, [PLAIN_ROW])
    csv_path.with_suffix(".parquet").write_bytes(b"existing")

    # Act
    convert.convert_to_parquet()

    # Assert
    assert csv_path.exists()
    assert csv_path.with_suffix(".parquet").read_bytes() == b"existing"
