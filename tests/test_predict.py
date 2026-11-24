"""
test_predict.py
---------------
Unit tests for customer churn inference module.
Uses standard library unittest.
"""

import os
import unittest
import pandas as pd
from src.modeling.predict import load_churn_model, predict_churn, assign_risk_tier

MODEL_PATH = "models/churn_model.pkl"
DATA_PATH = "data/processed/customer_churn_features.csv"


class TestPredictModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.path.exists(MODEL_PATH), f"Model not found at: {MODEL_PATH}"
        assert os.path.exists(DATA_PATH), f"Data not found at: {DATA_PATH}"
        cls.model = load_churn_model(MODEL_PATH)
        cls.sample_df = pd.read_csv(DATA_PATH).head(20)

    def test_load_model(self):
        """Verify model object has predict_proba method."""
        self.assertTrue(hasattr(self.model, "predict_proba"))

    def test_assign_risk_tier(self):
        """Verify risk tier assignment logic."""
        self.assertEqual(assign_risk_tier(0.85), "High Risk")
        self.assertEqual(assign_risk_tier(0.55), "Medium Risk")
        self.assertEqual(assign_risk_tier(0.20), "Low Risk")

    def test_predict_churn_output(self):
        """Verify inference output structure and ranges."""
        preds = predict_churn(self.sample_df, model_path=MODEL_PATH)
        self.assertEqual(len(preds), len(self.sample_df))
        self.assertIn("customer_id", preds.columns)
        self.assertIn("churn_probability", preds.columns)
        self.assertIn("churn_prediction", preds.columns)
        self.assertIn("risk_tier", preds.columns)

        # Value bounds
        self.assertTrue(((preds["churn_probability"] >= 0.0) & (preds["churn_probability"] <= 1.0)).all())
        self.assertTrue(set(preds["churn_prediction"].unique()).issubset({0, 1}))
        self.assertTrue(set(preds["risk_tier"].unique()).issubset({"High Risk", "Medium Risk", "Low Risk"}))


if __name__ == "__main__":
    unittest.main()
