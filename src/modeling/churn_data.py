"""
churn_data.py
==============
Temporal feature engineering and leakage-free dataset split for customer churn prediction.

E-Commerce Churn Definition:
----------------------------
A customer is considered churned (churn = 1) if they made at least one commercial
purchase in the historical observation window (T_start to T_cutoff) but placed
zero completed orders in the subsequent prediction window (T_cutoff to T_end).
Customers who placed one or more orders in the target window are active (churn = 0).

Strict Leakage-Free Design:
----------------------------
All predictive behavioral features (recency, frequency, spend, product variety,
cancellations, velocity, short-term momentum) are calculated STRICTLY from
transactions occurring prior to T_cutoff.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_cleaned_transactions(filepath: str) -> pd.DataFrame:
    """Load cleaned transaction records and filter for commercial merchandise."""
    logger.info(f"Loading cleaned transactions from: {filepath}")
    df = pd.read_csv(
        filepath,
        compression="gzip" if filepath.endswith(".gz") else None,
        parse_dates=["InvoiceDate"]
    )
    
    # Filter to registered merchandise transactions
    filtered = df[(df["Is_Registered"] == True) & (df["Item_Category"] == "Merchandise")].copy()
    logger.info(f"Loaded {len(filtered):,} registered merchandise rows across {filtered['Customer ID'].nunique():,} unique customers.")
    return filtered


def get_temporal_cutoff(df: pd.DataFrame, cutoff_days: int = 90) -> pd.Timestamp:
    """Determine the observation cutoff timestamp based on the target horizon."""
    max_date = df["InvoiceDate"].max()
    cutoff_date = max_date - pd.Timedelta(days=cutoff_days)
    logger.info(f"Dataset date range: {df['InvoiceDate'].min()} to {max_date}")
    logger.info(f"Observation Cutoff Date ({cutoff_days}-day target window): {cutoff_date}")
    return cutoff_date


def engineer_churn_features(df: pd.DataFrame, cutoff_days: int = 90) -> pd.DataFrame:
    """
    Engineer leakage-free customer features from the observation window
    and compute the ground-truth churn target from the prediction window.
    """
    cutoff_date = get_temporal_cutoff(df, cutoff_days=cutoff_days)
    
    # Split into observation and prediction horizons
    obs_df = df[df["InvoiceDate"] < cutoff_date].copy()
    target_df = df[df["InvoiceDate"] >= cutoff_date].copy()
    
    logger.info(f"Observation window transactions: {len(obs_df):,}")
    logger.info(f"Target window transactions: {len(target_df):,}")
    
    completed_obs = obs_df[obs_df["Is_Cancelled"] == False]
    cancelled_obs = obs_df[obs_df["Is_Cancelled"] == True]
    
    # Identify customers who placed completed orders in the target window (retained)
    target_active_custs = set(target_df[target_df["Is_Cancelled"] == False]["Customer ID"].unique())
    
    # Aggregate completed purchase behavior in observation window
    gb = completed_obs.groupby("Customer ID")
    invoices = gb["Invoice"].nunique()
    total_spend = gb["TotalAmount"].sum()
    total_qty = gb["Quantity"].sum()
    unique_items = gb["StockCode"].nunique()
    first_dt = gb["InvoiceDate"].min()
    last_dt = gb["InvoiceDate"].max()
    
    # Aggregate cancellation behavior
    gb_canc = cancelled_obs.groupby("Customer ID")
    canc_invoices = gb_canc["Invoice"].nunique()
    canc_amount = gb_canc["TotalAmount"].sum().abs()
    
    # Short-term momentum signals (30d and 60d prior to cutoff)
    m30 = completed_obs[completed_obs["InvoiceDate"] >= (cutoff_date - pd.Timedelta(days=30))].groupby("Customer ID")
    orders_30d = m30["Invoice"].nunique()
    spend_30d = m30["TotalAmount"].sum()
    
    m60 = completed_obs[completed_obs["InvoiceDate"] >= (cutoff_date - pd.Timedelta(days=60))].groupby("Customer ID")
    orders_60d = m60["Invoice"].nunique()
    spend_60d = m60["TotalAmount"].sum()
    
    # Domestic vs International geography (modal country)
    country_uk = completed_obs.groupby("Customer ID")["Country"].apply(
        lambda s: int((s == "United Kingdom").mode()[0]) if len(s) > 0 else 1
    )
    
    # Assemble feature matrix
    recency = (cutoff_date - last_dt).dt.total_seconds() / 86400.0
    tenure = (cutoff_date - first_dt).dt.total_seconds() / 86400.0
    span = (last_dt - first_dt).dt.total_seconds() / 86400.0
    
    feat_df = pd.DataFrame({
        "customer_id": invoices.index,
        "recency_days": recency.round(2),
        "frequency": invoices,
        "monetary_total": total_spend.round(2),
        "monetary_avg": (total_spend / invoices).round(2),
        "quantity_total": total_qty,
        "quantity_avg": (total_qty / invoices).round(2),
        "unique_products": unique_items,
        "tenure_days": tenure.round(2),
        "purchase_span_days": span.round(2),
        "cancellation_count": canc_invoices.reindex(invoices.index, fill_value=0),
        "cancellation_rate": (
            canc_invoices.reindex(invoices.index, fill_value=0) /
            (invoices + canc_invoices.reindex(invoices.index, fill_value=0))
        ).round(4),
        "refund_amount": canc_amount.reindex(invoices.index, fill_value=0.0).round(2),
        "orders_30d": orders_30d.reindex(invoices.index, fill_value=0),
        "spend_30d": spend_30d.reindex(invoices.index, fill_value=0.0).round(2),
        "orders_60d": orders_60d.reindex(invoices.index, fill_value=0),
        "spend_60d": spend_60d.reindex(invoices.index, fill_value=0.0).round(2),
        "is_uk": country_uk.reindex(invoices.index, fill_value=1),
        "churn": [int(cid not in target_active_custs) for cid in invoices.index]
    })
    
    # Derived order velocity (average days between repeat orders)
    feat_df["order_velocity_days"] = (
        feat_df["purchase_span_days"] / np.maximum(feat_df["frequency"] - 1, 1)
    ).round(2)
    
    # Momentum spend ratio (fraction of spend occurring in last 60 days)
    feat_df["momentum_spend_ratio"] = (
        feat_df["spend_60d"] / np.maximum(feat_df["monetary_total"], 1.0)
    ).clip(0.0, 1.0).round(4)
    
    logger.info(f"Engineered feature dataset: {feat_df.shape[0]:,} customers x {feat_df.shape[1]} columns.")
    churn_rate = feat_df["churn"].mean() * 100
    logger.info(f"Target Churn Distribution: {feat_df['churn'].sum():,} churned ({churn_rate:.2f}%), {(1 - feat_df['churn']).sum():,} retained ({100 - churn_rate:.2f}%).")
    
    return feat_df


def generate_churn_visualizations(feat_df: pd.DataFrame, output_dir: str):
    """Generate and save EDA charts for churn class balance and feature correlations."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    # 1. Class Distribution Chart
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    churn_counts = feat_df["churn"].value_counts()
    churn_labels = ["Churned (No return in 90d)", "Retained (Purchased in 90d)"]
    palette = ["#e74c3c", "#2ecc71"]
    
    bars = ax.bar([0, 1], [churn_counts.get(1, 0), churn_counts.get(0, 0)], color=palette, width=0.55, edgecolor="black", linewidth=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(churn_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Customer Count", fontsize=11, fontweight="bold")
    ax.set_title("Customer Churn Target Distribution (90-Day Observation Window)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(churn_counts) * 1.18)
    
    total = len(feat_df)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:,}\n({h/total*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )
    
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "churn_class_distribution.png")
    fig.savefig(chart1_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved churn class distribution plot to: {chart1_path}")
    
    # 2. Feature Correlation with Churn
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    numeric_cols = [c for c in feat_df.columns if c not in ["customer_id", "churn"]]
    corrs = feat_df[numeric_cols + ["churn"]].corr()["churn"].drop("churn").sort_values()
    
    colors = ["#2ecc71" if val < 0 else "#e74c3c" for val in corrs.values]
    bars = ax.barh(corrs.index, corrs.values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Pearson Correlation with Churn Target (churn = 1)", fontsize=11, fontweight="bold")
    ax.set_title("Feature Correlations with Customer Churn (Leakage-Free)", fontsize=13, fontweight="bold", pad=12)
    
    for bar in bars:
        w = bar.get_width()
        offset = 0.015 if w >= 0 else -0.015
        ha = "left" if w >= 0 else "right"
        ax.annotate(
            f"{w:+.3f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(offset * 100, 0),
            textcoords="offset points",
            ha=ha, va="center",
            fontsize=9, fontweight="bold"
        )
    
    ax.set_xlim(min(corrs.values) - 0.08, max(corrs.values) + 0.08)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "churn_feature_correlations.png")
    fig.savefig(chart2_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved churn correlation plot to: {chart2_path}")


def build_churn_dataset(
    input_path: str = "data/interim/online_retail_cleaned.csv.gz",
    output_path: str = "data/processed/customer_churn_features.csv",
    figures_dir: str = "reports/figures",
    cutoff_days: int = 90
) -> pd.DataFrame:
    """End-to-end execution pipeline for temporal churn dataset generation."""
    logger.info("=================================================================")
    logger.info("Starting Week 3: Temporal Churn Dataset Engineering Pipeline")
    logger.info("=================================================================")
    
    df = load_cleaned_transactions(input_path)
    feat_df = engineer_churn_features(df, cutoff_days=cutoff_days)
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat_df.to_csv(output_path, index=False)
    logger.info(f"Exported customer churn feature dataset to: {output_path}")
    
    # Generate EDA figures
    generate_churn_visualizations(feat_df, figures_dir)
    
    logger.info("Temporal churn dataset construction successfully completed.")
    return feat_df


if __name__ == "__main__":
    build_churn_dataset()
