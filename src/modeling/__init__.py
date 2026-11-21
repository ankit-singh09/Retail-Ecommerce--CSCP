"""
Modeling package for customer segmentation and churn prediction.
"""

from .segmentation import (
    load_rfm_features,
    preprocess_features,
    evaluate_kmeans_clusters,
    train_optimal_kmeans,
    assign_cluster_personas,
    run_customer_segmentation_pipeline,
)
from .churn_data import (
    build_churn_dataset,
    engineer_churn_features,
    load_cleaned_transactions,
)

__all__ = [
    "load_rfm_features",
    "preprocess_features",
    "evaluate_kmeans_clusters",
    "train_optimal_kmeans",
    "assign_cluster_personas",
    "run_customer_segmentation_pipeline",
    "build_churn_dataset",
    "engineer_churn_features",
    "load_cleaned_transactions",
]
