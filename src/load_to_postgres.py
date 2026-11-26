"""Load the retail data into a local PostgreSQL database.

Creates two tables and one view:
    retail_raw    original CSV, untouched
    retail_clean  cleaned data from src/dataset.py
    customer_rfm  view with one row per customer (src/sql/customer_rfm.sql)

Setup:
    1. Install PostgreSQL and create a database (for example: retail_db).
    2. Copy .env.example to .env and put your own password in it.
    3. Run from the repo root:  python src/load_to_postgres.py
"""

import io
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/online_retail_II.csv")
CLEAN_PATH = Path("data/interim/online_retail_cleaned.csv.gz")
SQL_DIR = Path("src/sql")
CHUNK_ROWS = 100_000

CLEAN_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
    "Item_Category": "item_category",
    "Is_Registered": "is_registered",
    "Is_Cancelled": "is_cancelled",
    "TotalAmount": "total_amount",
}

RAW_COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def get_engine():
    """Build a SQLAlchemy engine from DATABASE_URL (read from .env or the environment)."""
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL is not set. Copy .env.example to .env and fill in your password."
        )
    return create_engine(url)


def run_sql_file(engine, path: Path) -> None:
    """Execute every statement in a .sql file."""
    with engine.begin() as conn:
        conn.exec_driver_sql(path.read_text(encoding="utf-8"))
    logger.info(f"Ran {path.name}")


def copy_dataframe(engine, df: pd.DataFrame, table: str) -> None:
    """Bulk-load a DataFrame with PostgreSQL COPY (much faster than INSERT)."""
    columns = ", ".join(df.columns)
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            for start in range(0, len(df), CHUNK_ROWS):
                buffer = io.StringIO()
                df.iloc[start:start + CHUNK_ROWS].to_csv(buffer, index=False, header=False)
                buffer.seek(0)
                cur.copy_expert(
                    f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv)", buffer
                )
        raw.commit()
    finally:
        raw.close()
    logger.info(f"Loaded {len(df):,} rows into {table}")


def prepare_raw() -> pd.DataFrame:
    """Read the raw CSV and rename columns to match retail_raw."""
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())]
    # Customer IDs are floats in the CSV (13085.0); store them as text without the decimal.
    df["customer_id"] = df["customer_id"].astype("Int64").astype("string")
    return df


def prepare_clean() -> pd.DataFrame:
    """Read the cleaned data and keep the columns used by the SQL queries."""
    df = pd.read_csv(CLEAN_PATH, usecols=list(CLEAN_COLUMNS))
    df = df.rename(columns=CLEAN_COLUMNS)[list(CLEAN_COLUMNS.values())]
    df["customer_id"] = df["customer_id"].astype(str)
    return df


def main() -> None:
    engine = get_engine()
    run_sql_file(engine, SQL_DIR / "schema.sql")
    copy_dataframe(engine, prepare_raw(), "retail_raw")
    copy_dataframe(engine, prepare_clean(), "retail_clean")
    run_sql_file(engine, SQL_DIR / "customer_rfm.sql")

    with engine.connect() as conn:
        for name in ("retail_raw", "retail_clean", "customer_rfm"):
            count = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar()
            logger.info(f"{name}: {count:,} rows")


if __name__ == "__main__":
    main()
