"""
extract_data.py

Generates synthetic Australian energy consumption data.
Based on AEMO consumption patterns across five NEM states.

I'm using synthetic data here because AEMO API access requires
registration. The patterns are modelled from publicly available
AEMO historical data and match real consumption ranges closely.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)


def generate_energy_data(days_back=30, save_dir="data/raw"):
    """
    Generate hourly energy consumption records for five Australian states.

    Args:
        days_back: How many days of history to create (default 30).
        save_dir:  Where to write the raw CSV.

    Returns:
        DataFrame with columns: timestamp, state, consumption_mw,
        temperature, is_weekend.
    """
    logger.info(f"Generating {days_back} days of synthetic energy data...")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 5-minute intervals -- matches AEMO dispatch reporting frequency
    end_date = datetime.now().replace(second=0, microsecond=0)
    start_date = end_date - timedelta(days=days_back)
    timestamps = pd.date_range(start=start_date, end=end_date, freq="5min")

    # Approximate base loads drawn from public AEMO data (MW)
    states = ["NSW", "VIC", "QLD", "SA", "WA"]
    base_loads = {
        "NSW": 8000,   # largest grid, most industrial load
        "VIC": 5500,   # second largest NEM state
        "QLD": 6500,   # high AC load in summer
        "SA":  1500,   # smaller grid, high renewables share
        "WA":  2000,   # separate SWIS grid, included for comparison
    }

    records = []
    rng = np.random.default_rng(seed=42)  # reproducible output

    for state in states:
        base = base_loads[state]

        for ts in timestamps:
            hour = ts.hour
            month = ts.month
            is_weekend = ts.dayofweek >= 5

            # Daily shape: trough around 3 AM, peak around 2 PM
            hour_factor = 0.7 + 0.3 * (1 - abs(hour - 14) / 14)

            # Weekends run about 15% lighter (less commercial load)
            weekend_factor = 0.85 if is_weekend else 1.0

            # Seasonal shape: higher in summer (cooling) and winter (heating)
            seasonal_factor = 0.9 + 0.2 * (abs(month - 6) / 6)

            # Gaussian noise scaled to 3% of base load
            noise = rng.normal(0, base * 0.03)

            consumption = base * hour_factor * weekend_factor * seasonal_factor + noise

            # Simple temperature model correlated with season and time of day
            temp_base = 15 + 10 * (abs(month - 6) / 6)
            temp_daily = 5 * (1 - abs(hour - 14) / 14)
            temperature = temp_base + temp_daily + rng.normal(0, 2)

            records.append({
                "timestamp":      ts,
                "state":          state,
                "consumption_mw": max(0.0, consumption),
                "temperature":    round(float(temperature), 2),
                "is_weekend":     is_weekend,
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df):,} records across {len(states)} states")

    out_file = Path(save_dir) / f"energy_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Saved to {out_file}")

    return df


if __name__ == "__main__":
    generate_energy_data(days_back=30)
    print("\nData extraction complete.")