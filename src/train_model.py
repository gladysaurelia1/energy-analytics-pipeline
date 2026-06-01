"""
train_model.py

Trains one Random Forest model per Australian state.

Why per-state?
  Earlier I tried a single model with a state_encoded feature. It hit
  97% R2 but the feature importance told the real story: state_encoded
  accounted for 93% of the signal. The model had basically memorised
  "NSW ~ 8000 MW, SA ~ 1500 MW" and was doing almost no temporal
  reasoning at all. Separate models fix this -- each model only sees
  one state, so it has to actually learn the hour/day/season patterns
  to make good predictions.

Target performance (from documented results):
  Average test R2:  ~87%
  Average MAPE:     ~3.5%
  Train/test gap:   <5% (no meaningful overfitting)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# Features the model actually sees at prediction time.
# No lag features -- they caused data leakage in v1 (see experiments/).
MODEL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_business_hour",
    "temperature",
]


def load_processed_data(data_path="data/processed/processed_energy_data.csv"):
    logger.info(f"Loading processed data from {data_path}")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded {len(df):,} records")
    return df


def prepare_state_data(df, state, target="consumption_mw"):
    """
    Slice out one state, drop any NaN rows, and do a time-ordered split.

    Using time-order (not random shuffle) is important here -- a random
    split would let future values bleed into the training set, which
    would inflate test scores artificially.
    """
    logger.info(f"Preparing data for {state}...")

    sdf = df[df["state"] == state].copy()
    sdf = sdf.dropna(subset=MODEL_FEATURES + [target])
    sdf = sdf.replace([np.inf, -np.inf], np.nan).dropna()
    sdf = sdf.sort_values("timestamp").reset_index(drop=True)

    X = sdf[MODEL_FEATURES]
    y = sdf[target]

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logger.info(f"  {state}: {len(X_train):,} train / {len(X_test):,} test")
    return X_train, X_test, y_train, y_test


def train_state_model(X_train, y_train, state):
    """
    Fit a regularised Random Forest for one state.

    Hyperparameter rationale:
      max_depth=12        Shallow enough to prevent memorisation of noise.
      min_samples_leaf=15 Each leaf needs at least 15 observations,
                          which pushes the model toward broader patterns.
      max_features=sqrt   Standard variance-reduction trick for RF.
    """
    logger.info(f"  Training model for {state}...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=30,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_state_model(model, X_train, X_test, y_train, y_test, state):
    """Compute train and test metrics, flag the overfitting gap."""
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2  = r2_score(y_test,  test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

    # MAPE -- skip zeros to avoid division errors
    mask = y_test != 0
    test_mape = float(
        np.mean(np.abs((y_test[mask] - test_pred[mask]) / y_test[mask])) * 100
    )

    gap = train_r2 - test_r2

    logger.info(f"\n{state}:")
    logger.info(f"  Train R2 : {train_r2*100:.1f}%")
    logger.info(f"  Test  R2 : {test_r2*100:.1f}%")
    logger.info(f"  Gap      : {gap*100:.1f}%")
    logger.info(f"  MAE      : {test_mae:.1f} MW")
    logger.info(f"  MAPE     : {test_mape:.1f}%")

    return {
        "state":    state,
        "train_r2": train_r2,
        "test_r2":  test_r2,
        "gap":      gap,
        "mae":      test_mae,
        "rmse":     test_rmse,
        "mape":     test_mape,
    }


def save_models(all_models, model_dir="data/models"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out = Path(model_dir) / "energy_forecast_models.pkl"
    joblib.dump(all_models, out)
    logger.info(f"\nAll models saved to {out}")


def display_summary(metrics_list):
    """Print an overall summary table and flag any issues."""
    df = pd.DataFrame(metrics_list)

    avg_r2   = df["test_r2"].mean()
    avg_gap  = df["gap"].mean()
    avg_mae  = df["mae"].mean()
    avg_mape = df["mape"].mean()

    logger.info("\n" + "=" * 55)
    logger.info("SUMMARY")
    logger.info("=" * 55)
    logger.info(f"Average test R2 : {avg_r2*100:.1f}%")
    logger.info(f"Average gap     : {avg_gap*100:.1f}%")
    logger.info(f"Average MAE     : {avg_mae:.1f} MW")
    logger.info(f"Average MAPE    : {avg_mape:.1f}%")
    logger.info("")

    for _, row in df.iterrows():
        logger.info(
            f"  {row['state']}  R2={row['test_r2']*100:.1f}%  "
            f"gap={row['gap']*100:.1f}%  MAPE={row['mape']:.1f}%"
        )

    logger.info("")

    if avg_gap < 0.05:
        logger.info("No meaningful overfitting (gap < 5%).")
    elif avg_gap < 0.10:
        logger.info("Minor overfitting detected (gap 5-10%). Worth monitoring.")
    else:
        logger.info("Significant overfitting (gap > 10%). Review features.")

    if 0.80 <= avg_r2 <= 0.92:
        logger.info("Test accuracy is in the target range (80-92%).")
    elif avg_r2 > 0.92:
        logger.info("Test accuracy looks high -- double-check for data leakage.")
    else:
        logger.info("Test accuracy below 80%. May need more data or tuning.")


def train_pipeline():
    """Train and evaluate per-state models, then save everything."""
    logger.info("")
    logger.info("=" * 55)
    logger.info("TRAINING PIPELINE -- PER-STATE MODELS")
    logger.info("=" * 55)

    df = load_processed_data()
    states = sorted(df["state"].unique())

    all_models  = {}
    all_metrics = []

    for state in states:
        logger.info("\n" + "-" * 55)

        X_train, X_test, y_train, y_test = prepare_state_data(df, state)

        model = train_state_model(X_train, y_train, state)

        metrics = evaluate_state_model(
            model, X_train, X_test, y_train, y_test, state
        )

        all_models[state] = {
            "model":        model,
            "feature_cols": MODEL_FEATURES,
        }
        all_metrics.append(metrics)

    save_models(all_models)
    display_summary(all_metrics)

    return all_models, all_metrics


if __name__ == "__main__":
    train_pipeline()
    print("\nModel training complete.")