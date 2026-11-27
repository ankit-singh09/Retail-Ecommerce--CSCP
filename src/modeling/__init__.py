"""
Modeling package for customer segmentation and churn prediction.
"""

from .segmentation import (
    evaluate_cluster_range,
    fit_kmeans_segmentation,
)
from .churn_data import (
    build_churn_dataset,
    engineer_churn_features,
    load_cleaned_transactions,
)
from .predict import (
    load_churn_model,
    predict_churn,
    run_batch_inference,
)

__all__ = [
    "evaluate_cluster_range",
    "fit_kmeans_segmentation",
    "build_churn_dataset",
    "engineer_churn_features",
    "load_cleaned_transactions",
    "load_churn_model",
    "predict_churn",
    "run_batch_inference",
]
