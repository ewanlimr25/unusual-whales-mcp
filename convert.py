"""Convert Unusual Whales CSV exports to Parquet, or revert Parquet back to CSV.

Usage:
    python convert.py           # CSV → Parquet (deletes CSVs)
    python convert.py --revert  # Parquet → CSV (deletes Parquets)
"""

import argparse
from pathlib import Path

import duckdb

STOCKS_DIR = Path.home() / "Documents" / "Stocks"

FOLDER_MAP = {
    "options": "All Options",
    "darkpool": "Dark pool",
    "hotchains": "Hot Option Chains",
    "screener": "Stock Screener",
    "oi": "OI changes",
}


def _csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    # Quote/escape are set explicitly: the sniffer only samples the first rows,
    # and quoted fields (e.g. multi-value report_flags "{a,b}") can first appear
    # millions of rows in, after the sniffer has already decided "no quoting".
    duckdb.execute(
        f"COPY (SELECT * FROM read_csv_auto('{csv_path}', header=true, "
        f"quote='\"', escape='\"')) "
        f"TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def convert_to_parquet() -> None:
    converted = skipped = failed = 0
    for folder in FOLDER_MAP.values():
        data_dir = STOCKS_DIR / folder
        if not data_dir.exists():
            print(f"  [missing] {folder}/")
            continue
        for csv_path in sorted(data_dir.glob("*.csv")):
            parquet_path = csv_path.with_suffix(".parquet")
            if parquet_path.exists():
                print(f"  [skip]    {csv_path.name}")
                skipped += 1
                continue
            print(f"  [convert] {csv_path.name} ...", end=" ", flush=True)
            try:
                _csv_to_parquet(csv_path, parquet_path)
            except duckdb.Error as e:
                # Remove the partial file so the next run retries instead of skipping
                parquet_path.unlink(missing_ok=True)
                print(f"FAILED\n    {str(e).splitlines()[0]}")
                failed += 1
                continue
            csv_path.unlink()
            print("done")
            converted += 1
    print(f"\n{converted} converted, {skipped} skipped, {failed} failed.")


def revert_to_csv() -> None:
    reverted = skipped = 0
    for folder in FOLDER_MAP.values():
        data_dir = STOCKS_DIR / folder
        if not data_dir.exists():
            print(f"  [missing] {folder}/")
            continue
        for parquet_path in sorted(data_dir.glob("*.parquet")):
            csv_path = parquet_path.with_suffix(".csv")
            if csv_path.exists():
                print(f"  [skip]    {parquet_path.name}")
                skipped += 1
                continue
            print(f"  [revert]  {parquet_path.name} ...", end=" ", flush=True)
            duckdb.execute(
                f"COPY (SELECT * FROM read_parquet('{parquet_path}')) "
                f"TO '{csv_path}' (HEADER, DELIMITER ',')"
            )
            parquet_path.unlink()
            print("done")
            reverted += 1
    print(f"\n{reverted} reverted, {skipped} skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Unusual Whales CSV exports to Parquet (or revert)"
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Convert Parquet files back to CSV",
    )
    args = parser.parse_args()

    if args.revert:
        print(f"Reverting Parquet → CSV in {STOCKS_DIR} ...")
        revert_to_csv()
    else:
        print(f"Converting CSV → Parquet in {STOCKS_DIR} ...")
        convert_to_parquet()


if __name__ == "__main__":
    main()
