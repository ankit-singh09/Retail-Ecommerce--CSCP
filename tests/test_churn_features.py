"""
test_churn_features.py
-----------------------
Unit and integrity tests for customer churn feature engineering.
Uses standard library unittest for portability.
"""

import os
import unittest
import pandas as pd

CHURN_DATA_PATH = "data/processed/customer_churn_features.csv"


class TestChurnFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.path.exists(CHURN_DATA_PATH), f"Churn dataset not found at {CHURN_DATA_PATH}"
        cls.df = pd.read_csv(CHURN_DATA_PATH)

    def test_churn_dataset_shape(self):
        """Verify customer churn dataset is populated with expected dimensions."""
        self.assertGreater(len(self.df), 5000, f"Expected > 5000 customers, got {len(self.df)}")
        self.assertEqual(self.df.shape[1], 21, f"Expected 21 columns, got {self.df.shape[1]}")

    def test_no_null_values(self):
        """Verify there are zero missing/null values across all engineered features."""
        null_counts = self.df.isnull().sum()
        self.assertEqual(null_counts.sum(), 0, f"Found null values in dataset: {null_counts[null_counts > 0]}")

    def test_unique_customers(self):
        """Verify every row corresponds to a unique customer ID."""
        self.assertEqual(self.df["customer_id"].nunique(), len(self.df), "Customer IDs are not unique")

    def test_target_churn_values(self):
        """Verify target column contains only binary 0 and 1 values."""
        unique_targets = set(self.df["churn"].unique())
        self.assertEqual(unique_targets, {0, 1}, f"Unexpected target values: {unique_targets}")

    def test_feature_domains(self):
        """Verify feature values adhere to valid domain constraints."""
        self.assertTrue((self.df["recency_days"] >= 0).all(), "Recency days cannot be negative")
        self.assertTrue((self.df["frequency"] >= 1).all(), "Frequency must be at least 1")
        self.assertTrue((self.df["tenure_days"] >= 0).all(), "Tenure days cannot be negative")
        self.assertTrue(((self.df["cancellation_rate"] >= 0) & (self.df["cancellation_rate"] <= 1)).all(), "Cancellation rate must be between 0 and 1")
        self.assertTrue(((self.df["momentum_spend_ratio"] >= 0) & (self.df["momentum_spend_ratio"] <= 1)).all(), "Momentum spend ratio must be between 0 and 1")


if __name__ == "__main__":
    unittest.main()
