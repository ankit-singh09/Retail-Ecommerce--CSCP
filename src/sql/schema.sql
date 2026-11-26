-- Schema for the Online Retail II project (PostgreSQL).
-- retail_raw   : the original CSV, loaded untouched (one row per invoice line).
-- retail_clean : output of src/dataset.py (duplicates removed, flags added).

DROP VIEW  IF EXISTS customer_rfm;
DROP TABLE IF EXISTS retail_clean;
DROP TABLE IF EXISTS retail_raw;

CREATE TABLE retail_raw (
    invoice      TEXT,
    stock_code   TEXT,
    description  TEXT,
    quantity     INTEGER,
    invoice_date TIMESTAMP,
    price        NUMERIC(12, 2),
    customer_id  TEXT,          -- NULL for guest checkouts
    country      TEXT
);

CREATE TABLE retail_clean (
    invoice       TEXT        NOT NULL,
    stock_code    TEXT,
    description   TEXT,
    quantity      INTEGER,
    invoice_date  TIMESTAMP   NOT NULL,
    price         NUMERIC(12, 2),
    customer_id   TEXT        NOT NULL,   -- real ID, or GUEST_<invoice> for guests
    country       TEXT,
    item_category TEXT,
    is_registered BOOLEAN     NOT NULL,
    is_cancelled  BOOLEAN     NOT NULL,
    total_amount  NUMERIC(14, 2)
);

CREATE INDEX idx_retail_clean_customer ON retail_clean (customer_id);
CREATE INDEX idx_retail_clean_date     ON retail_clean (invoice_date);
