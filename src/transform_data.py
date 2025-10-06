"""
Data Transformation Module
Cleans raw data and engineers features for ML
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_latest_raw_data(data_dir='data/raw'):
    """Load the most recent raw data file"""
    raw_dir = Path(data_dir)
    csv_files = list(raw_dir.glob('energy_data_*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No raw data files found in {data_dir}")
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading {latest_file}")
    
    df = pd.read_csv(latest_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def clean_data(df):
    """
    Remove duplicates, handle missing values, remove outliers
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Cleaning data: {len(df):,} records")
    
    initial_count = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp', 'state'])
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['timestamp', 'state', 'consumption_mw'])
    
    # Remove outliers using IQR method (per state)
    df_clean = []
    
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        
        # Calculate quartiles
        Q1 = state_df['consumption_mw'].quantile(0.25)
        Q3 = state_df['consumption_mw'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds (3x IQR is lenient for energy data)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Filter outliers
        state_df = state_df[
            (state_df['consumption_mw'] >= lower_bound) & 
            (state_df['consumption_mw'] <= upper_bound)
        ]
        
        df_clean.append(state_df)
    
    df = pd.concat(df_clean, ignore_index=True)
    
    removed = initial_count - len(df)
    logger.info(f"✓ Removed {removed:,} records during cleaning")
    
    return df

def engineer_features(df):
    """
    Create features for machine learning model
    
    Args:
        df: Cleaned DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")
    
    df = df.copy()
    df = df.sort_values(['state', 'timestamp'])
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_month'] = df['timestamp'].dt.day
    df['is_business_hour'] = df['hour'].between(9, 17).astype(int)
    
    # Lag features (previous consumption values)
    # 12 intervals = 1 hour (5-min intervals)
    df['consumption_lag_1h'] = df.groupby('state')['consumption_mw'].shift(12)
    
    # 288 intervals = 24 hours
    df['consumption_lag_24h'] = df.groupby('state')['consumption_mw'].shift(288)
    
    # Rolling statistics (1-hour window)
    df['consumption_rolling_mean_1h'] = df.groupby('state')['consumption_mw'].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean()
    )
    
    df['consumption_rolling_std_1h'] = df.groupby('state')['consumption_mw'].transform(
        lambda x: x.rolling(window=12, min_periods=1).std()
    )
    
    logger.info(f"✓ Feature engineering complete. Shape: {df.shape}")
    
    return df

def save_processed_data(df, output_dir='data/processed'):
    """Save processed data to CSV"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / 'processed_energy_data.csv'
    
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved processed data to {output_file}")
    
    return output_file

def transform_pipeline():
    """Execute the full transformation pipeline"""
    logger.info("=" * 60)
    logger.info("DATA TRANSFORMATION PIPELINE")
    logger.info("=" * 60)
    
    # Load data
    df = load_latest_raw_data()
    
    # Clean data
    df = clean_data(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Save processed data
    save_processed_data(df)
    
    logger.info("=" * 60)
    logger.info("✓ Transformation complete!")
    logger.info("=" * 60)
    
    return df

if __name__ == "__main__":
    transform_pipeline()
    print("\nData transformation complete!")