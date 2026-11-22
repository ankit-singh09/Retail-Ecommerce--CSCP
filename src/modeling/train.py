"""
Model training pipeline for customer churn prediction.

Trains Logistic Regression and Random Forest on the leakage-free churn
features, compares them with cross-validation, evaluates on a held-out
test set, and saves the best model (by test ROC-AUC) to models/.

Run from the repo root:  python src/modeling/train.py
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/processed/customer_churn_features.csv"
MODEL_PATH = "models/churn_model.pkl"
REPORT_PATH = "reports/churn_model_metrics.md"
RANDOM_STATE = 42


def load_data(path: str = DATA_PATH):
    """Load churn features and split into X (features) and y (churn target)."""
    df = pd.read_csv(path)
    y = df["churn"]
    X = df.drop(columns=["customer_id", "churn"])
    return X, y


def build_models() -> dict:
    """Return the candidate models. Class weights handle class imbalance."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                )),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    """Compute Precision, Recall, F1 and ROC-AUC on the test set."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }


def train() -> dict:
    """Train both models, report metrics, and save the best one to models/."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    for name, model in build_models().items():
        # Cross-validation runs on the training split only, so the test set stays untouched.
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics["CV ROC-AUC (mean)"] = cv_auc.mean()
        results[name], fitted[name] = metrics, model
        logger.info(f"{name}: {metrics}")

    best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)
    logger.info(f"Saved best model ({best_name}) to {MODEL_PATH}")

    report = pd.DataFrame(results).T.round(3)
    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REPORT_PATH).write_text(
        "# Churn Model Metrics\n\n"
        f"Best model: **{best_name}** (target: ROC-AUC >= 0.75)\n\n"
        + report.to_markdown() + "\n"
    )
    return results


if __name__ == "__main__":
    train()