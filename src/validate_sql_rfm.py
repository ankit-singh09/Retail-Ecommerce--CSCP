"""Check that the SQL customer_rfm view matches the Python RFM table.

Run after src/load_to_postgres.py and src/features.py:
    python src/validate_sql_rfm.py
"""

import pandas as pd

from load_to_postgres import get_engine

PYTHON_RFM = "data/processed/customer_rfm_features.csv"

# (SQL column, Python column, tolerance)
CHECKS = [
    ("frequency", "Frequency", 0),
    ("monetary", "Monetary", 0.011),
    ("recency_days", "Recency_Days", 0),
    ("tenure_days", "Tenure_Days", 0),
    ("avg_order_value", "Avg_Order_Value", 0.011),
    ("unique_products", "Unique_Products", 0),
    ("cancelled_invoices", "Cancelled_Invoices", 0),
    ("cancellation_rate", "Cancellation_Rate", 0.00011),
]


def main() -> None:
    sql_df = pd.read_sql("SELECT * FROM customer_rfm", get_engine())
    sql_df["customer_id"] = sql_df["customer_id"].astype(str)
    py_df = pd.read_csv(PYTHON_RFM)
    py_df["Customer ID"] = py_df["Customer ID"].astype(str)

    print(f"SQL customers: {len(sql_df):,} | Python customers: {len(py_df):,}")
    merged = sql_df.merge(
        py_df, left_on="customer_id", right_on="Customer ID", how="outer", indicator=True
    )
    print("Customer IDs in both / only SQL / only Python:",
          (merged["_merge"] == "both").sum(),
          (merged["_merge"] == "left_only").sum(),
          (merged["_merge"] == "right_only").sum())

    both = merged[merged["_merge"] == "both"]
    all_ok = len(sql_df) == len(py_df) == len(both)
    for sql_col, py_col, tol in CHECKS:
        diff = (both[sql_col].astype(float) - both[py_col].astype(float)).abs()
        bad = int((diff > tol).sum())
        all_ok &= bad == 0
        print(f"{py_col:<20} mismatches: {bad}")
    print("RESULT:", "SQL matches Python" if all_ok else "DIFFERENCES FOUND")


if __name__ == "__main__":
    main()
