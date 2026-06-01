"""
main.py

Pipeline orchestrator for the Australian energy analytics project.
Runs three steps in sequence: extract -> transform -> train.

Usage:
    python src/main.py           # default 30 days
    python src/main.py --days 60
    python src/main.py --days 7  # quick smoke test
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Make sure sibling modules are importable when called from the project root
sys.path.insert(0, str(Path(__file__).parent))

from extract_data import generate_energy_data
from transform_data import (
    load_latest_raw_data,
    clean_data,
    engineer_features,
    save_processed_data,
)
from train_model import train_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def step_1_extract(days_back):
    print(f"\n{'='*65}")
    print(f"  STEP 1/3: DATA EXTRACTION")
    print(f"{'='*65}\n")

    try:
        df = generate_energy_data(days_back=days_back)
        logger.info(f"Generated {len(df):,} records")
        logger.info(f"States : {', '.join(sorted(df['state'].unique()))}")
        logger.info(f"Range  : {df['timestamp'].min()} to {df['timestamp'].max()}")
        return True
    except Exception as exc:
        logger.error(f"Step 1 failed: {exc}")
        return False


def step_2_transform():
    print(f"\n{'='*65}")
    print(f"  STEP 2/3: DATA TRANSFORMATION")
    print(f"{'='*65}\n")

    try:
        df = load_latest_raw_data()
        logger.info(f"Loaded {len(df):,} raw records")

        df = clean_data(df)
        logger.info(f"After cleaning: {len(df):,} records")

        df = engineer_features(df)
        logger.info(f"Features built: {len(df.columns)} columns total")

        out = save_processed_data(df)
        logger.info(f"Saved to {out}")
        return True
    except Exception as exc:
        logger.error(f"Step 2 failed: {exc}")
        return False


def step_3_train():
    print(f"\n{'='*65}")
    print(f"  STEP 3/3: MODEL TRAINING")
    print(f"{'='*65}\n")

    try:
        train_pipeline()
        return True
    except Exception as exc:
        logger.error(f"Step 3 failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def print_success():
    print(f"\n{'='*65}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*65}")
    print("\nOutput files:")
    print("  data/raw/energy_data_*.csv")
    print("  data/processed/processed_energy_data.csv")
    print("  data/models/energy_forecast_models.pkl")
    print("\nNext step:")
    print("  streamlit run src/dashboard.py")
    print(f"{'='*65}\n")
    logger.info("Pipeline completed successfully.")


def print_failure(step_name):
    print(f"\n{'='*65}")
    print(f"  PIPELINE FAILED: {step_name}")
    print(f"{'='*65}")
    print("\nTroubleshooting:")
    print("  1. Check the error message above")
    print("  2. pip install -r requirements.txt")
    print("  3. Check pipeline.log for the full trace")
    print(f"{'='*65}\n")
    logger.error(f"Pipeline failed at: {step_name}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(days_back=30):
    logger.info(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 65)

    if not step_1_extract(days_back):
        print_failure("Step 1: Data Extraction")
        return False

    if not step_2_transform():
        print_failure("Step 2: Data Transformation")
        return False

    if not step_3_train():
        print_failure("Step 3: Model Training")
        return False

    print_success()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Australian Energy Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py
  python src/main.py --days 60
  python src/main.py --days 7
        """,
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data to generate (default: 30, min: 7)",
    )
    args = parser.parse_args()

    if args.days < 7:
        logger.error("Minimum 7 days required for a meaningful train/test split.")
        sys.exit(1)
    if args.days > 365:
        logger.warning("Generating more than 365 days will take several minutes.")

    success = run(days_back=args.days)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()