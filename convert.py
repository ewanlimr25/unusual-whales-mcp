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


def convert_to_parquet() -> None:
    converted = skipped = 0
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
            duckdb.execute(
                f"COPY (SELECT * FROM read_csv_auto('{csv_path}', header=true)) "
                f"TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            csv_path.unlink()
            print("done")
            converted += 1
    print(f"\n{converted} converted, {skipped} skipped.")


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
