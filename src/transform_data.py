"""
transform_data.py

Cleans the raw energy CSV and builds features for model training.

Feature choices:
  - Time features (hour, day_of_week, month, is_business_hour) capture
    the cyclical consumption patterns that actually drive forecast accuracy.
  - Temperature is included because HVAC load is the main driver of
    consumption spikes in both summer and winter.
  - Lag features are intentionally excluded -- they caused severe
    overfitting in earlier experiments (see experiments/README.md).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)


def load_latest_raw_data(data_dir="data/raw"):
    """Load the most recently modified raw energy CSV."""
    raw_dir = Path(data_dir)
    csv_files = list(raw_dir.glob("energy_data_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No raw data files found in {data_dir}")

    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading {latest.name}")

    df = pd.read_csv(latest)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean_data(df):
    """
    Remove duplicates, drop rows missing critical values,
    and filter outliers per state using a 3x IQR rule.

    3x IQR is intentionally lenient -- energy grids have real spikes
    (heat waves, industrial load changes) that a tighter rule would
    incorrectly remove.
    """
    logger.info(f"Cleaning {len(df):,} raw records...")
    initial = len(df)

    df = df.drop_duplicates(subset=["timestamp", "state"])
    df = df.dropna(subset=["timestamp", "state", "consumption_mw"])

    cleaned_states = []
    for state in df["state"].unique():
        sdf = df[df["state"] == state].copy()

        q1 = sdf["consumption_mw"].quantile(0.25)
        q3 = sdf["consumption_mw"].quantile(0.75)
        iqr = q3 - q1

        sdf = sdf[
            (sdf["consumption_mw"] >= q1 - 3 * iqr) &
            (sdf["consumption_mw"] <= q3 + 3 * iqr)
        ]
        cleaned_states.append(sdf)

    df = pd.concat(cleaned_states, ignore_index=True)
    logger.info(f"Removed {initial - len(df):,} records. {len(df):,} remain.")
    return df


def engineer_features(df):
    """
    Build the feature set used during model training.

    Time features are straightforward extractions from the timestamp.
    is_business_hour is a simple flag for 9-17 weekday hours -- it
    adds signal beyond hour alone because commercial load switches on
    and off at those boundaries.

    Lag and rolling features are created here (they live in the
    processed CSV) but are NOT passed to the model. They were kept
    in the output for EDA purposes only.
    """
    logger.info("Building features...")

    df = df.copy().sort_values(["state", "timestamp"])

    # Core time features
    df["hour"]            = df["timestamp"].dt.hour
    df["day_of_week"]     = df["timestamp"].dt.dayofweek
    df["month"]           = df["timestamp"].dt.month
    df["day_of_month"]    = df["timestamp"].dt.day
    df["is_business_hour"] = df["hour"].between(9, 17).astype(int)

    # Lag features -- stored for EDA, excluded from model training
    # 12 x 5-min intervals = 1 hour; 288 = 24 hours
    df["consumption_lag_1h"] = (
        df.groupby("state")["consumption_mw"].shift(12)
    )
    df["consumption_lag_24h"] = (
        df.groupby("state")["consumption_mw"].shift(288)
    )
    df["consumption_rolling_mean_1h"] = df.groupby("state")["consumption_mw"].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean()
    )
    df["consumption_rolling_std_1h"] = df.groupby("state")["consumption_mw"].transform(
        lambda x: x.rolling(window=12, min_periods=1).std()
    )

    logger.info(f"Feature engineering done. Output shape: {df.shape}")
    return df


def save_processed_data(df, output_dir="data/processed"):
    """Write the processed DataFrame to CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / "processed_energy_data.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved processed data to {out}")
    return out


def transform_pipeline():
    """Run the full transformation pipeline end to end."""
    logger.info("=" * 55)
    logger.info("DATA TRANSFORMATION PIPELINE")
    logger.info("=" * 55)

    df = load_latest_raw_data()
    df = clean_data(df)
    df = engineer_features(df)
    save_processed_data(df)

    logger.info("Transformation complete.")
    return df


if __name__ == "__main__":
    transform_pipeline()
    print("\nData transformation complete.")