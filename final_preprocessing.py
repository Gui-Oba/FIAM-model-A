"""Preprocess monthly Parquet files by ranking features to [-1, 1]."""

from pathlib import Path
import sys
from typing import List

import pandas as pd


CSV_FEATURES = Path("factor_char_list.csv")
PARQUETS_DIR = Path("final_monthly_parquets")
OUTPUT_DIR = Path("final_preprocessed_parquets")
TARGET = "stock_ret"


def _load_feature_list(csv_path: Path) -> List[str]:
    """Return list of feature names from the CSV."""
    df = pd.read_csv(csv_path)
    return df["variable"].astype(str).tolist()


def _rank_to_unit_range(series: pd.Series) -> pd.Series:
    """Rank a series to [-1, 1] after median imputation."""
    s = series.copy()
    median = s.median(skipna=True)
    s = s.fillna(median)
    # Dense rank: 1..K -> shift to 0..K-1
    s = s.rank(method="dense") - 1
    smax = s.max()
    if pd.isna(smax) or smax <= 0:
        return pd.Series(0.0, index=s.index)
    # Scale to [-1, 1]
    return (s / smax) * 2.0 - 1.0


def preprocessing() -> None:
    """Read monthly Parquet files and write processed versions."""
    pd.set_option("mode.chained_assignment", None)  # disable copy warnings

    if not PARQUETS_DIR.exists():
        print(f"Error: missing directory {PARQUETS_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = _load_feature_list(CSV_FEATURES)

    parquet_files = sorted(p for p in PARQUETS_DIR.iterdir() if p.suffix == ".parquet")

    for pfile in parquet_files:
        print(f"Processing {pfile.name}")
        group = pd.read_parquet(pfile)

        # Ensure datetime type
        if "ret_eom" in group.columns:
            group["ret_eom"] = pd.to_datetime(group["ret_eom"])

        # Keep rows with valid target
        group = group[group[TARGET].notna()].copy()

        # Rank-transform feature columns
        for var in features:
            if var in group.columns:
                group[var] = _rank_to_unit_range(group[var])
            else:
                print(f"Warning: {var} not found in {pfile.name}")

        out_path = OUTPUT_DIR / pfile.name
        group.to_parquet(out_path, index=False)
        print(f"Saved {out_path}")

    print(f"Finished processing {len(parquet_files)} Parquet files.")


if __name__ == "__main__":
    preprocessing()