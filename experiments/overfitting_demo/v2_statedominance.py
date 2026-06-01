"""
experiments/v2_statedominance.py

Second attempt -- fixed the data leakage by removing lag features,
but introduced a different problem: state_encoded dominated everything.

R2 improved from 99.4% (fake) to 97.0% (still too high), and MAPE
went from 3% to 11% -- the model had actually gotten worse at the task
it was supposed to do, even though the headline metric looked cleaner.

Feature importance breakdown:
  state_encoded  93%
  hour            2%
  everything else < 5% combined

The model was essentially memorising "NSW ~8000 MW, SA ~1500 MW" and
applying a small temporal correction on top. It wasn't learning daily
or seasonal patterns in any meaningful sense.

Fix: train one model per state (see train_model.py). With only one
state in the training data, the model has to actually learn the
temporal patterns -- there's nothing else to learn from.

This file is kept as a reference for the progression of experiments.
Do not use it in production.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
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
    No lag features this time, but state_encoded is still included as
    a single feature for all states. This is the problematic setup.
    """
    logger.info("Preparing features (no lag features, but single model for all states)")

    feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "day_of_month",
        "is_business_hour",
        "temperature",
    ]

    if "state" in df.columns:
        df["state_encoded"] = pd.Categorical(df["state"]).codes
        feature_cols.append("state_encoded")

    available = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available)} features: {available}")

    df_ml = df.dropna(subset=available + [target]).copy()
    df_ml = df_ml.replace([np.inf, -np.inf], np.nan).dropna()
    df_ml = df_ml.sort_values("timestamp")

    X = df_ml[available]
    y = df_ml[target]

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    logger.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, available


def train_model(X_train, y_train):
    logger.info("Training Random Forest (single model, all states)...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
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

    logger.info("\nFeature importance:")
    for _, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    logger.info(
        "\n  NOTE: state_encoded will likely dominate (>90%). "
        "This means the model is mostly learning state-level offsets, "
        "not temporal patterns."
    )

    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2  = r2_score(y_test,  test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

    mask = y_test != 0
    test_mape = float(
        np.mean(np.abs((y_test[mask] - test_pred[mask]) / y_test[mask])) * 100
    )

    gap = train_r2 - test_r2

    logger.info(f"\nTrain R2 : {train_r2*100:.2f}%")
    logger.info(f"Test  R2 : {test_r2*100:.2f}%")
    logger.info(f"Gap      : {gap*100:.2f}%")
    logger.info(f"MAE      : {test_mae:.2f} MW")
    logger.info(f"MAPE     : {test_mape:.2f}%")

    return {
        "train_R2": train_r2,
        "test_R2":  test_r2,
        "MAE":      test_mae,
        "RMSE":     test_rmse,
        "MAPE":     test_mape,
        "gap":      gap,
    }


def save_model(model, feature_cols, model_dir="data/models"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out = Path(model_dir) / "energy_forecast_model_v2_statedom.pkl"
    joblib.dump({"model": model, "feature_cols": feature_cols}, out)
    logger.info(f"Saved to {out}")


def train_pipeline():
    logger.info("=" * 55)
    logger.info("V2 -- STATE DOMINANCE MODEL (experiment only)")
    logger.info("=" * 55)

    df = load_processed_data()
    X_train, X_test, y_train, y_test, feature_cols = prepare_ml_data(df)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    save_model(model, feature_cols)

    return model, metrics


if __name__ == "__main__":
    train_pipeline()
    print("\nV2 experiment complete.")