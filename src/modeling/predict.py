"""
predict.py
----------
Inference module for customer churn prediction.

Loads the trained churn model artifact and generates churn probabilities,
binary churn classifications, and business risk tiers for customer records.

Usage:
    python src/modeling/predict.py
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/churn_model.pkl"
DEFAULT_DATA_PATH = "data/processed/customer_churn_features.csv"
DEFAULT_OUTPUT_PATH = "data/processed/customer_churn_predictions.csv"


def load_churn_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load serialized model artifact from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    logger.info(f"Loading trained churn model from: {model_path}")
    model = joblib.load(model_path)
    return model


def assign_risk_tier(prob: float) -> str:
    """Categorize churn probability into actionable business risk tiers."""
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def predict_churn(
    df: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Generate churn predictions and probability scores for given features.

    Parameters:
        df: DataFrame containing customer features.
        model_path: Path to serialized model file.
        threshold: Decision threshold for binary classification.

    Returns:
        DataFrame with customer_id (if available), churn_prob, churn_pred, and risk_tier.
    """
    model = load_churn_model(model_path)

    # Exclude non-feature identifier / target columns
    cols_to_drop = [c for c in ["customer_id", "churn"] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Generate probabilities
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    results = pd.DataFrame({
        "churn_probability": np.round(proba, 4),
        "churn_prediction": preds,
        "risk_tier": [assign_risk_tier(p) for p in proba]
    })

    if "customer_id" in df.columns:
        results.insert(0, "customer_id", df["customer_id"].values)

    return results


def run_batch_inference(
    input_path: str = DEFAULT_DATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Run batch inference over dataset and export predictions."""
    logger.info(f"Reading features from: {input_path}")
    df = pd.read_csv(input_path)

    preds_df = predict_churn(df, model_path=model_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preds_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(preds_df):,} batch predictions to: {output_path}")

    # Log tier distribution
    tier_counts = preds_df["risk_tier"].value_counts()
    for tier, count in tier_counts.items():
        logger.info(f"  {tier}: {count:,} ({count / len(preds_df) * 100:.1f}%)")

    return preds_df


if __name__ == "__main__":
    run_batch_inference()
