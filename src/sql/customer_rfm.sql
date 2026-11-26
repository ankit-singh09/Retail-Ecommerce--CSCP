-- One row per registered customer: Recency, Frequency, Monetary and related
-- metrics. This mirrors build_rfm_features() in src/features.py.
--
-- Rules (same as the Python code):
--   * registered customers only (guest rows have no stable customer)
--   * merchandise only (no postage, discounts or gift vouchers)
--   * Frequency counts completed (non-cancelled) invoices
--   * Monetary is net spend: completed purchases minus cancelled/returned lines
--   * keep customers with at least 1 completed invoice and net spend > 0.01
--   * Recency = days from the customer's last purchase to (last date in data + 1)

CREATE OR REPLACE VIEW customer_rfm AS
WITH base AS (
    SELECT *
    FROM retail_clean
    WHERE is_registered
      AND item_category = 'Merchandise'
),
snapshot AS (
    SELECT (MAX(invoice_date)::date + 1) AS snapshot_date
    FROM base
),
net AS (
    SELECT customer_id,
           ROUND(SUM(total_amount), 2) AS monetary
    FROM base
    GROUP BY customer_id
),
completed AS (
    SELECT customer_id,
           COUNT(DISTINCT invoice)     AS frequency,
           ROUND(SUM(total_amount), 2) AS gross_monetary,
           MIN(invoice_date)           AS first_purchase,
           MAX(invoice_date)           AS last_purchase,
           COUNT(DISTINCT stock_code)  AS unique_products
    FROM base
    WHERE NOT is_cancelled
    GROUP BY customer_id
),
cancelled AS (
    SELECT customer_id,
           COUNT(DISTINCT invoice) AS cancelled_invoices
    FROM base
    WHERE is_cancelled
    GROUP BY customer_id
)
SELECT
    n.customer_id,
    c.frequency,
    n.monetary,
    (s.snapshot_date - c.last_purchase::date)                  AS recency_days,
    (c.last_purchase::date - c.first_purchase::date)           AS tenure_days,
    ROUND(c.gross_monetary / c.frequency, 2)                   AS avg_order_value,
    c.unique_products,
    COALESCE(x.cancelled_invoices, 0)                          AS cancelled_invoices,
    ROUND(COALESCE(x.cancelled_invoices, 0)::numeric
          / (c.frequency + COALESCE(x.cancelled_invoices, 0)), 4) AS cancellation_rate,
    c.first_purchase,
    c.last_purchase
FROM net n
JOIN completed c USING (customer_id)
LEFT JOIN cancelled x USING (customer_id)
CROSS JOIN snapshot s
WHERE c.frequency >= 1
  AND n.monetary > 0.01;
