"""
Main Pipeline Orchestrator
Runs the complete energy analytics pipeline
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extract_data import generate_energy_data
from transform_data import load_latest_raw_data, clean_data, engineer_features, save_processed_data
from train_model import train_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def print_step_header(step_num, step_name):
    """Print formatted step header"""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}/3: {step_name}")
    print(f"{'='*70}\n")

def step_1_extract_data(days_back=30):
    """Step 1: Extract energy consumption data"""
    print_step_header(1, "DATA EXTRACTION")
    
    try:
        logger.info(f"Generating {days_back} days of energy data...")
        df = generate_energy_data(days_back=days_back)
        
        logger.info(f"Successfully generated {len(df):,} records")
        logger.info(f"States: {', '.join(sorted(df['state'].unique()))}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")
        return False

def step_2_transform_data():
    """Step 2: Transform and clean data"""
    print_step_header(2, "DATA TRANSFORMATION")
    
    try:
        logger.info("Loading raw data...")
        df = load_latest_raw_data()
        logger.info(f"Loaded {len(df):,} raw records")
        
        logger.info("Cleaning data...")
        df = clean_data(df)
        logger.info(f"Cleaned data: {len(df):,} records remaining")
        
        logger.info("Engineering features...")
        df = engineer_features(df)
        logger.info(f"Created {len(df.columns)} total features")
        
        logger.info("Saving processed data...")
        output_path = save_processed_data(df)
        logger.info(f"Saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data transformation failed: {str(e)}")
        return False

def step_3_train_models():
    """Step 3: Train ML models"""
    print_step_header(3, "MODEL TRAINING")
    
    try:
        logger.info("Training Random Forest models per state...")
        models, metrics = train_pipeline()
        logger.info("All models trained successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False

def print_success_summary():
    """Print completion message"""
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files:")
    print("  - data/raw/energy_data_*.csv")
    print("  - data/processed/processed_energy_data.csv")
    print("  - data/models/energy_forecast_models.pkl")
    print("\nNext steps:")
    print("  1. Launch dashboard: streamlit run src/dashboard.py")
    print("  2. View at: http://localhost:8501")
    print("="*70 + "\n")
    
    logger.info("Pipeline execution completed successfully")

def print_failure_summary(failed_step):
    """Print failure message"""
    print("\n" + "="*70)
    print(f"PIPELINE FAILED AT: {failed_step}")
    print("="*70)
    print("\nTroubleshooting steps:")
    print("  1. Check the error message above")
    print("  2. Verify dependencies: pip install -r requirements.txt")
    print("  3. Check log file: pipeline.log")
    print("="*70 + "\n")
    
    logger.error(f"Pipeline failed at: {failed_step}")

def run_full_pipeline(days_back=30):
    """
    Execute the complete energy analytics pipeline
    
    Args:
        days_back: Number of days of historical data to generate
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting energy analytics pipeline")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    # Step 1: Extract Data
    if not step_1_extract_data(days_back):
        print_failure_summary("Step 1: Data Extraction")
        return False
    
    # Step 2: Transform Data
    if not step_2_transform_data():
        print_failure_summary("Step 2: Data Transformation")
        return False
    
    # Step 3: Train Models
    if not step_3_train_models():
        print_failure_summary("Step 3: Model Training")
        return False
    
    # Success
    print_success_summary()
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Australian Energy Analytics Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                    # Run with default 30 days
  python src/main.py --days 60          # Generate 60 days of data
  python src/main.py --days 7           # Quick test with 7 days
        """
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days of historical data to generate (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if args.days < 7:
        logger.error("Error: Minimum 7 days required for meaningful training")
        sys.exit(1)
    
    if args.days > 365:
        logger.warning("Warning: Generating >365 days will take longer")
    
    # Run pipeline
    success = run_full_pipeline(days_back=args.days)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
