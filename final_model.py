"""Expanding-window model training and evaluation on monthly parquet data."""

from pathlib import Path
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


PARQUETS_DIR = Path("final_preprocessed_parquets")
FACTOR_LIST_CSV = Path("factor_char_list.csv")
TARGET = "stock_ret"
START_DATE = pd.to_datetime("20050101", format="%Y%m%d")
END_DATE = pd.to_datetime("20260101", format="%Y%m%d")
OUT_CSV = Path("output.csv")


def filter_features_advanced(
    data: pd.DataFrame,
    features: List[str],
    completeness_threshold: float = 0.7,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
) -> List[str]:
    """Select features by completeness, variance, and correlation pruning."""
    valid = []
    for feat in features:
        if feat in data.columns:
            non_missing = data[feat].notna().mean()
            if non_missing >= completeness_threshold:
                var = data[feat].var(skipna=True)
                if pd.notna(var) and var > variance_threshold:
                    valid.append(feat)

    if len(valid) <= 1:
        return valid

    corr = data[valid].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > correlation_threshold).any()]
    final = [f for f in valid if f not in to_drop]
    print(
        "Advanced filtering: %d → %d → %d features"
        % (len(features), len(valid), len(final))
    )
    print("Dropped highly correlated:", len(to_drop))
    return final


def load_data_from_parquets(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    period_name: str,
) -> pd.DataFrame:
    """Load monthly parquet files in [start_date, end_date)."""
    print(
        "Loading %s data from %s to %s"
        % (period_name, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    )

    if not PARQUETS_DIR.exists():
        print("Parquet directory not found:", PARQUETS_DIR)
        return pd.DataFrame()

    data_parts = []
    files_loaded = 0

    for fname in os.listdir(PARQUETS_DIR):
        if not fname.endswith(".parquet"):
            continue
        date_str = fname.replace("data_", "").replace(".parquet", "")
        try:
            fdate = pd.to_datetime(f"{date_str}-01", format="%Y-%m-%d")
        except ValueError:
            print("Skip unparsable filename:", fname)
            continue

        if start_date <= fdate < end_date:
            path = PARQUETS_DIR / fname
            try:
                monthly = pd.read_parquet(path)
            except Exception as err:  # pylint: disable=broad-except
                print("Failed to read %s: %s" % (fname, err))
                continue
            data_parts.append(monthly)
            files_loaded += 1

    if not data_parts:
        print("No %s data found" % period_name)
        return pd.DataFrame()

    combined = pd.concat(data_parts, ignore_index=True)
    print("%s dataset: %d files, %d rows" % (period_name, files_loaded, len(combined)))
    return combined


def compute_cutoffs(counter: int) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (train_start, val_start, test_start, test_end) for the loop index."""
    train_start = START_DATE + pd.DateOffset(years=counter)
    val_start = START_DATE + pd.DateOffset(years=8 + counter)
    test_start = START_DATE + pd.DateOffset(years=10 + counter)
    test_end = START_DATE + pd.DateOffset(years=11 + counter)
    return train_start, val_start, test_start, test_end


def model() -> None:
    """Run expanding-window training and save predictions."""
    if not FACTOR_LIST_CSV.exists():
        raise FileNotFoundError(f"Missing feature list: {FACTOR_LIST_CSV}")

    stock_vars = list(pd.read_csv(FACTOR_LIST_CSV)["variable"].astype(str).values)
    print("Starting expanding-window analysis")
    print(
        "Window from %s to %s"
        % (START_DATE.strftime("%Y-%m-%d"), END_DATE.strftime("%Y-%m-%d"))
    )

    counter = 0
    pred_out = pd.DataFrame()

    while (START_DATE + pd.DateOffset(years=11 + counter)) <= END_DATE:
        print("=" * 60)
        print("Iteration:", counter + 1)
        train_start, val_start, test_start, test_end = compute_cutoffs(counter)

        print("Training:   %s to %s" % (train_start.date(), val_start.date()))
        print("Validation: %s to %s" % (val_start.date(), test_start.date()))
        print("Testing:    %s to %s" % (test_start.date(), test_end.date()))

        train = load_data_from_parquets(train_start, val_start, "training")
        validate = load_data_from_parquets(val_start, test_start, "validation")
        test = load_data_from_parquets(test_start, test_end, "testing")

        if train.empty or validate.empty or test.empty:
            print("Skip iteration: empty dataset")
            counter += 1
            continue

        valid_vars = filter_features_advanced(
            train, stock_vars, completeness_threshold=0.7, variance_threshold=0.01
        )

        x_train = train[valid_vars].values
        y_train = train[TARGET].values
        x_val = validate[valid_vars].values  # kept for future model selection
        _ = x_val  # silence unused warning without removing validation stage
        x_test = test[valid_vars].values
        y_test = test[TARGET].values
        _ = y_test  # not used in-place, evaluation is on accumulated preds

        # Simple engineered interactions
        mom_idx = [i for i, n in enumerate(valid_vars) if "mom" in n.lower()]
        val_idx = [
            i
            for i, n in enumerate(valid_vars)
            if any(t in n.lower() for t in ["pe", "pb", "btm"])
        ]

        new_train = []
        new_test = []

        if mom_idx and val_idx:
            new_train.append(x_train[:, mom_idx[0]] * x_train[:, val_idx[0]])
            new_test.append(x_test[:, mom_idx[0]] * x_test[:, val_idx[0]])

        if len(mom_idx) >= 2:
            new_train.append(x_train[:, mom_idx[0]] * x_train[:, mom_idx[1]])
            new_test.append(x_test[:, mom_idx[0]] * x_test[:, mom_idx[1]])

        if new_train:
            x_train = np.column_stack([x_train] + new_train)
            x_test = np.column_stack([x_test] + new_test)
            print("Added %d interaction features. Train shape: %s" % (len(new_train), x_train.shape))

        # Target de-mean for no-intercept fit
        y_mean = float(np.mean(y_train))
        y_train_dm = y_train - y_mean
        print("y_train mean: %.6f" % y_mean)

        # XGBoost regressor
        reg = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        reg.fit(x_train, y_train_dm)
        y_pred = reg.predict(x_test) + y_mean

        reg_pred = test[["year", "month", "ret_eom", "id", TARGET]].copy()
        reg_pred["xgb"] = y_pred
        pred_out = pd.concat([pred_out, reg_pred], ignore_index=True)

        # Rolling R2 on accumulated predictions
        y_real = pred_out[TARGET].values
        r2 = 1.0 - np.sum((y_real - pred_out["xgb"].values) ** 2) / np.sum(y_real ** 2)
        print("Accumulated R2: %.6f" % r2)

        counter += 1
        print("Completed iteration", counter)

    print("=" * 60)
    print("Analysis complete. Iterations:", counter)

    pred_out.to_csv(OUT_CSV, index=False)
    print("Saved predictions:", OUT_CSV)

    # Final R2
    y_real = pred_out[TARGET].values
    r2 = 1.0 - np.sum((y_real - pred_out["xgb"].values) ** 2) / np.sum(y_real ** 2)
    print("Final R2: %.6f" % r2)


if __name__ == "__main__":
    model()