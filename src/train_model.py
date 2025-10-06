"""
ML model training module
Trains separate models for each state to avoid statet encoding dominance
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_data(data_path='data/processed/processed_energy_data.csv'):
    """Load processed data"""
    logger.info(f"Loading processed data from {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"✓ Loaded {len(df):,} records")
    return df

def prepare_ml_data_per_state(df, state, target='consumption_mw'):
    """
    Prepare features for a SINGLE state
    This way the model learns temporal patterns, not state differences
    """
    logger.info(f"\nPreparing data for {state}...")
    
    # Filter to single state
    df_state = df[df['state'] == state].copy()
    
    # Features that capture temporal patterns
    feature_cols = [
        'hour',
        'day_of_week',
        'month',
        'is_business_hour',
        'temperature'
    ]
    
    # Clean data
    df_ml = df_state.dropna(subset=feature_cols + [target]).copy()
    df_ml = df_ml.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Sort by time
    df_ml = df_ml.sort_values('timestamp')
    
    X = df_ml[feature_cols]
    y = df_ml[target]
    
    # Time-based split
    train_size = int(0.8 * len(X))
    
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    logger.info(f"  Training: {len(X_train):,} | Testing: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_single_state_model(X_train, y_train, state):
    """Train model for one state"""
    logger.info(f"  Training model for {state}...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,           # Even more regularization
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_single_model(model, X_train, X_test, y_train, y_test, state):
    """Evaluate single state model"""
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    mask = y_test != 0
    test_mape = np.mean(np.abs((y_test[mask] - test_pred[mask]) / y_test[mask])) * 100
    
    gap = train_r2 - test_r2
    
    logger.info(f"\n{state} Results:")
    logger.info(f"  Train R²: {train_r2:.4f} ({train_r2*100:.2f}%)")
    logger.info(f"  Test R²:  {test_r2:.4f} ({test_r2*100:.2f}%)")
    logger.info(f"  Gap:      {gap:.4f} ({gap*100:.2f}%)")
    logger.info(f"  MAE:      {test_mae:.2f} MW")
    logger.info(f"  MAPE:     {test_mape:.2f}%")
    
    return {
        'state': state,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': gap,
        'mae': test_mae,
        'rmse': test_rmse,
        'mape': test_mape
    }

def train_pipeline():
    """Train separate model for each state"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE - PER-STATE MODELS")
    logger.info("=" * 60)
    
    df = load_processed_data()
    
    states = sorted(df['state'].unique())
    all_models = {}
    all_metrics = []
    
    for state in states:
        logger.info("\n" + "-" * 60)
        
        # Prepare data for this state
        X_train, X_test, y_train, y_test, feature_cols = prepare_ml_data_per_state(df, state)
        
        # Train model
        model = train_single_state_model(X_train, y_train, state)
        
        # Evaluate
        metrics = evaluate_single_model(model, X_train, X_test, y_train, y_test, state)
        
        # Store
        all_models[state] = {
            'model': model,
            'feature_cols': feature_cols
        }
        all_metrics.append(metrics)
    
    # Save all models
    save_models(all_models)
    
    # Summary
    display_summary(all_metrics)
    
    return all_models, all_metrics

def save_models(all_models, model_dir='data/models'):
    """Save all state models"""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir) / 'energy_forecast_models.pkl'
    
    joblib.dump(all_models, model_path)
    logger.info(f"\n✓ All models saved to {model_path}")

def display_summary(all_metrics):
    """Display overall performance summary"""
    logger.info("\n" + "=" * 60)
    logger.info("OVERALL PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    df_metrics = pd.DataFrame(all_metrics)
    
    logger.info(f"\nAverage Test R²:  {df_metrics['test_r2'].mean():.4f} ({df_metrics['test_r2'].mean()*100:.2f}%)")
    logger.info(f"Average Gap:      {df_metrics['gap'].mean():.4f} ({df_metrics['gap'].mean()*100:.2f}%)")
    logger.info(f"Average MAE:      {df_metrics['mae'].mean():.2f} MW")
    logger.info(f"Average MAPE:     {df_metrics['mape'].mean():.2f}%")
    
    logger.info("\nPer-state Summary:")
    logger.info("-" * 60)
    for _, row in df_metrics.iterrows():
        logger.info(f"{row['state']:3s}: Test R²={row['test_r2']:.3f} | Gap={row['gap']:.3f} | MAPE={row['mape']:.1f}%")
    
    # Overfitting check
    avg_gap = df_metrics['gap'].mean()
    if avg_gap < 0.05:
        logger.info("\n✓ Minimal overfitting across all models (avg gap < 5%)")
    elif avg_gap < 0.10:
        logger.info("\n! Some overfitting detected (avg gap < 10%)")
    else:
        logger.info("\n!!!Significant overfitting (avg gap > 10%)")
    
    # Realistic check
    avg_test_r2 = df_metrics['test_r2'].mean()
    if 0.80 <= avg_test_r2 <= 0.92:
        logger.info("✓ excellent performance")
    elif 0.65 <= avg_test_r2 < 0.80:
        logger.info("✓ Good performance")
    elif avg_test_r2 > 0.92:
        logger.info("! Results suspiciously high - verify no data leakage")
    else:
        logger.info("! Performance below standard - needs improvement")

if __name__ == "__main__":
    train_pipeline()
    print("\nModel training complete!")