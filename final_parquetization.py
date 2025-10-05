"""Convert a large CSV into month-partitioned Parquet files."""

from pathlib import Path
from typing import Dict

import pandas as pd


CSV_PATH = Path("ret_sample.csv")
PARQUETS_DIR = Path("final_monthly_parquets")
CHUNK_SIZE = 100_000


def parquetization() -> None:
    """Stream CSV by chunks and write monthly Parquet files."""
    PARQUETS_DIR.mkdir(parents=True, exist_ok=True)
    print("Step 1: Converting CSV to monthly Parquet files...")

    month_counter: Dict[pd.Period, int] = {}

    reader = pd.read_csv(
        CSV_PATH,
        chunksize=CHUNK_SIZE,
        parse_dates=["date", "ret_eom", "char_date", "char_eom"],
        low_memory=False,
    )

    for chunk_num, chunk in enumerate(reader, start=1):
        print(f"Processing chunk {chunk_num}...")

        # Group by month using ret_eom
        chunk["year_month"] = chunk["ret_eom"].dt.to_period("M")

        for year_month, month_df in chunk.groupby("year_month"):
            filename = PARQUETS_DIR / f"data_{year_month}.parquet"

            # Drop helper column before saving
            data_to_write = month_df.drop(columns="year_month")

            if filename.exists():
                existing = pd.read_parquet(filename)
                combined = pd.concat([existing, data_to_write], ignore_index=True)
                combined.to_parquet(filename, index=False)
            else:
                data_to_write.to_parquet(filename, index=False)

            month_counter[year_month] = month_counter.get(year_month, 0) + len(month_df)

    print(f"Created {len(month_counter)} monthly Parquet files")
    print(f"Complete. All Parquet files saved in: {PARQUETS_DIR}")


if __name__ == "__main__":
    parquetization()
