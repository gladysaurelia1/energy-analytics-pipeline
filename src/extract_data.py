"""
Energy data extraction module
Generate synthetic australian energy consumption data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_energy_data(days_back=30, save_dir='data/raw'):
    """
    Generate synthetic australian energy consumption data
    
    Args:
        days_back: Number of days of historical data to generate
        save_dir: Directory to save raw data
    
    Returns:
        DataFrame with generated energy data
    """
    logger.info(f"Generating {days_back} days of energy data...")
    
    # Create output directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamps (5-minute intervals)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Australian states with typical base load (MW)
    states = ['NSW', 'VIC', 'QLD', 'SA', 'WA']
    base_loads = {
        'NSW': 8000,   # New South Wales (most populous)
        'VIC': 5500,   # Victoria
        'QLD': 6500,   # Queensland
        'SA': 1500,    # South Australia
        'WA': 2000     # Western Australia
    }
    
    data = []
    
    for state in states:
        base = base_loads[state]
        
        for ts in timestamps:
            # Daily pattern: peak at 2 PM, low at 3 AM
            hour = ts.hour
            hour_factor = 0.7 + 0.3 * (1 - abs(hour - 14) / 14)
            
            # Weekly pattern: lower consumption on weekends
            is_weekend = ts.dayofweek >= 5
            weekend_factor = 0.85 if is_weekend else 1.0
            
            # Seasonal pattern (summer/winter peaks)
            month = ts.month
            seasonal_factor = 0.9 + 0.2 * (abs(month - 6) / 6)
            
            # Add realistic random noise
            noise = np.random.normal(0, base * 0.03)
            
            # Calculate consumption
            consumption = base * hour_factor * weekend_factor * seasonal_factor + noise
            
            # Simulate temperature (affects energy usage)
            temp_base = 15 + 10 * (abs(month - 6) / 6)  # Warmer in summer/winter
            temp_daily = 5 * (abs(hour - 14) / 14)      # Warmer midday
            temperature = temp_base + temp_daily + np.random.normal(0, 2)
            
            data.append({
                'timestamp': ts,
                'state': state,
                'consumption_mw': max(0, consumption),  # No negative consumption
                'temperature': temperature,
                'is_weekend': is_weekend
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    logger.info(f"✓ Generated {len(df):,} records for {len(states)} states")
    
    # Save to CSV
    output_file = Path(save_dir) / f'energy_data_{datetime.now().strftime("%Y%m%d")}.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # to generate data
    generate_energy_data(days_back=30)
    print("\n Data extraction complete!")