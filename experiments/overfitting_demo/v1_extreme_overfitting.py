"""
experiments/v1_extreme_overfitting.py

First attempt at training a consumption forecast model.
Achieved 99.4% R2 on the test set, which should have been an
immediate warning sign. It wasn't -- I spent a day thinking
I'd just built a great model before realising the numbers
were too clean to be real.

The problem: lag features (consumption_lag_1h, consumption_lag_24h)
effectively hand the model the answer. If you ask "what will
consumption be at 14:05?" and the model can see "what was it at 14:00?",
it barely needs to learn anything -- it just returns the lag value
with a small correction. This is classic data leakage for time series.

Evidence:
  consumption_rolling_mean_1h  ~41% importance
  consumption_lag_1h           ~24%
  consumption_lag_24h          ~22%
  hour                          ~0.5%

Temporal features like hour and day_of_week should drive a
consumption model. When they show near-zero importance, something
is wrong. The lag features were doing all the heavy lifting.

See README.md in the experiments folder for the full story.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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


def load_processed_data(data_path="data/processed/processed_energy_data.csv"):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded {len(df):,} records")
    return df


def prepare_ml_data(df, target="consumption_mw"):
    """
    Build features including lag values.
    This is the problematic setup -- kept here as a reference.
    """
    logger.info("Preparing features (includes lag features -- data leakage version)")

    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_business_hour",
        "temperature",
        "consumption_lag_1h",      # leaks future info into past-based features
        "consumption_lag_24h",
        "consumption_rolling_mean_1h",
    ]

    available = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available)} features: {available}")

    df_ml = df.dropna(subset=available + [target]).copy()
    df_ml = df_ml.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info(f"Records after cleaning: {len(df_ml):,}")

    if len(df_ml) < 1000:
        raise ValueError(f"Not enough clean data: only {len(df_ml)} rows remain.")

    if "state" in df_ml.columns:
        df_ml["state_encoded"] = pd.Categorical(df_ml["state"]).codes
        available.append("state_encoded")

    X = df_ml[available]
    y = df_ml[target]

    # Random shuffle -- also incorrect for time series, compounds the leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    logger.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, available


def train_model(X_train, y_train):
    logger.info("Training Random Forest (overfitted configuration)...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,           # too deep -- memorises training data
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Training done.")

    importance = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("\nTop 5 features by importance:")
    for _, row in importance.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating...")

    pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2   = r2_score(y_test, pred)

    mask = y_test != 0
    mape = float(np.mean(np.abs((y_test[mask] - pred[mask]) / y_test[mask])) * 100)

    logger.info(f"\n  MAE  : {mae:.2f} MW")
    logger.info(f"  RMSE : {rmse:.2f} MW")
    logger.info(f"  R2   : {r2:.4f} ({r2*100:.2f}%)")
    logger.info(f"  MAPE : {mape:.2f}%")
    logger.info("\n  NOTE: These numbers look great. They are not.")
    logger.info("  Lag features are leaking the answer into the input.")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "predictions": pred}


def save_model(model, feature_cols, model_dir="data/models"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out = Path(model_dir) / "energy_forecast_model_v1_overfitted.pkl"
    joblib.dump({"model": model, "feature_cols": feature_cols}, out)
    logger.info(f"Saved to {out}")


def train_pipeline():
    logger.info("=" * 55)
    logger.info("V1 -- OVERFITTED MODEL (experiment only)")
    logger.info("=" * 55)

    df = load_processed_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_ml_data(df)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, feature_cols)

    return model, metrics


if __name__ == "__main__":
    train_pipeline()
    print("\nV1 experiment complete.")