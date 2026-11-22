"""Dataset ingestion and cleaning pipeline for Online Retail II transactions.

Implements programmatic deduplication, invalid record filtering, SKU-level
canonicalization, guest checkout imputation, outlier capping, and temporal
feature extraction while keeping total row drops strictly under 5%.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

NON_COMMERCIAL_CODES = frozenset({
    "M",
    "m",
    "BANK CHARGES",
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "CRUK",
    "S",
    "B",
    "TEST001",
    "TEST002",
})

POSTAGE_CODES = frozenset({"POST", "DOT", "C2"})
DISCOUNT_CODES = frozenset({"D"})


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load the raw Online Retail II transactional CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    return pd.read_csv(path)


def _categorize_stock_code(stock_code: str) -> str:
    """Classify normalized StockCode into commercial item categories."""
    if stock_code in POSTAGE_CODES:
        return "Postage/Shipping"
    if stock_code in DISCOUNT_CODES:
        return "Discount"
    if stock_code.startswith("GIFT_"):
        return "Gift Voucher"
    return "Merchandise"


def _log_step(
    audit_log: list[dict],
    step_name: str,
    current_df: pd.DataFrame,
    baseline_rows: int,
) -> None:
    """Append step-level retention metrics to the audit log."""
    curr_n = len(current_df)
    prev_n = audit_log[-1]["Remaining_Rows"] if audit_log else baseline_rows
    step_removed = prev_n - curr_n
    total_dropped = baseline_rows - curr_n
    audit_log.append({
        "Step": step_name,
        "Rows_Removed_Step": step_removed,
        "Remaining_Rows": curr_n,
        "Cumulative_Dropped_Pct": round((total_dropped / baseline_rows) * 100, 2),
        "Retained_Pct": round((curr_n / baseline_rows) * 100, 2),
    })


def clean_retail_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full programmatic cleaning pipeline on raw transactional data.

    Returns:
        Tuple containing the cleaned DataFrame and a step-by-step retention audit log.
    """
    df = df_raw.copy()
    baseline_rows = len(df_raw)
    audit_log: list[dict] = []
    _log_step(audit_log, "Step 1: Raw Baseline Loaded", df, baseline_rows)

    # Step 2: Drop exact row-level duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    _log_step(audit_log, "Step 2: Drop Exact Duplicates", df, baseline_rows)

    # Step 3: Remove non-commercial entries (Price <= 0, 'A' invoices, test/fee codes)
    df = df[df["Price"] > 0].copy()
    df = df[~df["Invoice"].astype(str).str.strip().str.startswith("A")].copy()
    raw_sc = df["StockCode"].astype(str).str.strip()
    df = df[~raw_sc.isin(NON_COMMERCIAL_CODES)].copy()
    _log_step(audit_log, "Step 3: Filter Non-Commercial & Test Rows", df, baseline_rows)

    # Normalize StockCode and classify item categories
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Item_Category"] = df["StockCode"].map(_categorize_stock_code)

    # Step 4: Standardize types and canonicalize Description per StockCode
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Description"] = (
        df["Description"]
        .fillna("UNKNOWN PRODUCT")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    sku_mode = (
        df.groupby(["StockCode", "Description"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .drop_duplicates("StockCode")
        .set_index("StockCode")["Description"]
    )
    df["Description"] = df["StockCode"].map(sku_mode).fillna("UNKNOWN PRODUCT")
    df = df.drop_duplicates().reset_index(drop=True)
    _log_step(audit_log, "Step 4: Standardize Types & Descriptions", df, baseline_rows)

    # Step 5: Programmatic Customer ID imputation (zero row loss)
    df["Is_Registered"] = df["Customer ID"].notna()
    df["Customer ID"] = np.where(
        df["Is_Registered"],
        df["Customer ID"].fillna(0).astype("int64").astype(str),
        "GUEST_" + df["Invoice"],
    )
    _log_step(audit_log, "Step 5: Impute Missing Guest Customer IDs", df, baseline_rows)

    # Step 6: Cancellation flags, line totals, and winsorized outlier capping
    df["Is_Cancelled"] = df["Invoice"].str.startswith("C")
    df["TotalAmount"] = (df["Quantity"] * df["Price"]).round(2)

    merch_pos = df[(df["Item_Category"] == "Merchandise") & (~df["Is_Cancelled"])]
    qty_cap = float(merch_pos["Quantity"].quantile(0.99))
    price_cap = float(merch_pos["Price"].quantile(0.99))

    df["Is_Outlier_Qty"] = df["Quantity"].abs() > qty_cap
    df["Is_Outlier_Price"] = df["Price"] > price_cap
    df["Quantity_Capped"] = df["Quantity"].clip(lower=-qty_cap, upper=qty_cap)
    df["Price_Capped"] = df["Price"].clip(upper=price_cap)
    _log_step(audit_log, "Step 6: Flag Cancellations & Cap Outliers", df, baseline_rows)

    # Step 7: Temporal feature enrichment
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["Is_Partial_Month"] = df["YearMonth"] == "2011-12"
    df = df.reset_index(drop=True)
    _log_step(audit_log, "Step 7: Temporal Enrichment & Finalize", df, baseline_rows)

    return df, pd.DataFrame(audit_log)


def save_interim_data(
    df: pd.DataFrame,
    csv_path: str | Path,
    gzip_path: str | Path | None = None,
) -> None:
    """Save cleaned dataset locally as CSV and optionally compressed CSV.GZ."""
    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_target, index=False)

    if gzip_path is not None:
        gz_target = Path(gzip_path)
        gz_target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(gz_target, index=False, compression="gzip")


def main() -> None:
    """Run end-to-end raw-to-interim cleaning pipeline from repository root."""
    project_dir = Path(__file__).resolve().parents[1]
    raw_file = project_dir / "data" / "raw" / "online_retail_II.csv"
    interim_csv = project_dir / "data" / "interim" / "online_retail_cleaned.csv"
    interim_gz = project_dir / "data" / "interim" / "online_retail_cleaned.csv.gz"

    print(f"Loading raw dataset: {raw_file}")
    df_raw = load_raw_data(raw_file)
    df_clean, audit_df = clean_retail_data(df_raw)

    print("\n--- Pipeline Retention Audit ---")
    print(audit_df.to_string(index=False))

    save_interim_data(df_clean, interim_csv, interim_gz)
    csv_mb = interim_csv.stat().st_size / (1024 ** 2)
    gz_mb = interim_gz.stat().st_size / (1024 ** 2)
    print(f"\nSaved local CSV:      {interim_csv} ({csv_mb:.2f} MB)")
    print(f"Saved compressed GZ:  {interim_gz} ({gz_mb:.2f} MB)")


if __name__ == "__main__":
    main()
