"""Customer-level RFM and behavioral feature engineering for Online Retail II."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _assign_rfm_segment(row: pd.Series) -> str:
    """Map R_Score and combined F/M score to standard retail RFM segments."""
    r = int(row["R_Score"])
    fm = (int(row["F_Score"]) + int(row["M_Score"])) / 2.0

    if r >= 4 and fm >= 4.0:
        return "Champions"
    if r >= 3 and fm >= 3.5:
        return "Loyal Customers"
    if r >= 4 and 2.0 <= fm < 4.0:
        return "Potential Loyalists"
    if r == 5 and fm < 2.0:
        return "Recent Customers"
    if r == 4 and fm < 2.0:
        return "Promising"
    if r == 3 and 2.0 <= fm < 3.5:
        return "Need Attention"
    if r <= 2 and fm >= 4.0:
        return "Cant Lose Them"
    if r <= 2 and 2.5 <= fm < 4.0:
        return "At Risk"
    if r == 2 and fm < 2.5:
        return "Hibernating"
    return "Lost"


def build_rfm_features(
    df_clean: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    merchandise_only: bool = True,
) -> pd.DataFrame:
    """Compute customer-level RFM metrics, scores, segments, and behavioral features.

    Args:
        df_clean: Cleaned interim transactional DataFrame from `src.dataset`.
        snapshot_date: Reference anchor date for Recency. Defaults to max date + 1 day.
        merchandise_only: If True, restrict to `Item_Category == 'Merchandise'`.

    Returns:
        DataFrame indexed sequentially with one row per valid registered customer.
    """
    df = df_clean.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Isolate longitudinal registered customers (exclude per-invoice guest IDs)
    df = df[df["Is_Registered"]].copy()
    if merchandise_only and "Item_Category" in df.columns:
        df = df[df["Item_Category"] == "Merchandise"].copy()

    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    completed = df[~df["Is_Cancelled"]].copy()
    cancelled = df[df["Is_Cancelled"]].copy()

    # Aggregate net activity across both completed and returned merchandise lines
    net_agg = df.groupby("Customer ID").agg(
        Net_Monetary=("TotalAmount", "sum"),
        Net_Units=("Quantity", "sum"),
        Total_Invoices=("Invoice", "nunique"),
    )
    net_agg["Net_Monetary"] = net_agg["Net_Monetary"].round(2)

    # Aggregate completed purchase behavior
    comp_agg = completed.groupby("Customer ID").agg(
        Frequency=("Invoice", "nunique"),
        Gross_Monetary=("TotalAmount", "sum"),
        Gross_Units=("Quantity", "sum"),
        First_Purchase=("InvoiceDate", "min"),
        Last_Purchase=("InvoiceDate", "max"),
        Unique_Products=("StockCode", "nunique"),
    )
    comp_agg["Gross_Monetary"] = comp_agg["Gross_Monetary"].round(2)

    # Aggregate cancellation counts & return values
    cancel_agg = cancelled.groupby("Customer ID").agg(
        Cancelled_Invoices=("Invoice", "nunique"),
        Return_Amount=("TotalAmount", lambda s: round(abs(float(s.sum())), 2)),
    )

    # Primary country per registered customer (mode)
    primary_country = (
        df.groupby(["Customer ID", "Country"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .drop_duplicates("Customer ID")
        .set_index("Customer ID")["Country"]
    )

    rfm = net_agg.join(comp_agg, how="inner").join(cancel_agg, how="left")
    rfm["Cancelled_Invoices"] = rfm["Cancelled_Invoices"].fillna(0).astype(int)
    rfm["Return_Amount"] = rfm["Return_Amount"].fillna(0.0)
    rfm["Country"] = rfm.index.map(primary_country)
    rfm["Is_UK"] = rfm["Country"] == "United Kingdom"

    # Keep active customers with positive net spending (eliminates fully-cancelled dust)
    rfm = rfm[(rfm["Frequency"] >= 1) & (rfm["Net_Monetary"] > 0.01)].copy()

    # Core RFM & behavioral metrics
    rfm["Recency_Days"] = (
        snapshot_date - rfm["Last_Purchase"].dt.normalize()
    ).dt.days
    rfm["Monetary"] = rfm["Net_Monetary"]
    rfm["Tenure_Days"] = (
        rfm["Last_Purchase"].dt.normalize() - rfm["First_Purchase"].dt.normalize()
    ).dt.days
    rfm["Avg_Order_Value"] = (rfm["Gross_Monetary"] / rfm["Frequency"]).round(2)
    rfm["Cancellation_Rate"] = (
        rfm["Cancelled_Invoices"] / (rfm["Frequency"] + rfm["Cancelled_Invoices"])
    ).round(4)

    # Quantile-based 1-5 RFM scoring (rank ties deterministically to avoid bin errors)
    rfm["R_Score"] = pd.qcut(
        rfm["Recency_Days"].rank(method="first"),
        q=5,
        labels=[5, 4, 3, 2, 1],
    ).astype(int)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    rfm["RFM_Score_Sum"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    rfm["RFM_Segment"] = rfm.apply(_assign_rfm_segment, axis=1)

    # Log-transformed features for right-skewed K-Means clustering
    rfm["Recency_Log"] = np.log1p(rfm["Recency_Days"]).round(4)
    rfm["Frequency_Log"] = np.log1p(rfm["Frequency"]).round(4)
    rfm["Monetary_Log"] = np.log1p(rfm["Monetary"]).round(4)

    rfm = rfm.reset_index()

    ordered_cols = [
        "Customer ID",
        "Country",
        "Is_UK",
        "Recency_Days",
        "Frequency",
        "Monetary",
        "Gross_Monetary",
        "Return_Amount",
        "Net_Units",
        "Gross_Units",
        "Avg_Order_Value",
        "Unique_Products",
        "Cancelled_Invoices",
        "Cancellation_Rate",
        "Tenure_Days",
        "First_Purchase",
        "Last_Purchase",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Score_Sum",
        "RFM_Segment",
        "Recency_Log",
        "Frequency_Log",
        "Monetary_Log",
    ]
    return rfm[ordered_cols]


def build_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper returning customer-level RFM features."""
    return build_rfm_features(df_clean)


def save_rfm_features(rfm_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist engineered customer RFM features to CSV."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rfm_df.to_csv(target, index=False)
    return target


def main() -> None:
    """Run interim-to-RFM feature engineering pipeline from repository root."""
    project_dir = Path(__file__).resolve().parents[1]
    interim_gz = project_dir / "data" / "interim" / "online_retail_cleaned.csv.gz"
    rfm_out = project_dir / "data" / "processed" / "customer_rfm_features.csv"

    print(f"Loading cleaned interim dataset: {interim_gz}")
    df_clean = pd.read_csv(interim_gz)
    rfm_df = build_rfm_features(df_clean)

    save_rfm_features(rfm_df, rfm_out)
    size_kb = rfm_out.stat().st_size / 1024
    print(f"Engineered RFM table shape: {rfm_df.shape[0]:,} customers x {rfm_df.shape[1]} features")
    print(f"Saved RFM features CSV:     {rfm_out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
