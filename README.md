# Retail E-commerce: Customer Segmentation & Churn Prediction

Axlero Data Analytics internship, Project 1. Segments retail customers with RFM and K-Means, then predicts which customers are likely to churn.

**Authors:** Ankit Singh, Aman Singh
**Dataset:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UK online retailer, Dec 2009 to Dec 2011, 1,067,371 invoice lines)
**Structure:** [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

## Business problem

Customer acquisition is expensive and retention is unpredictable. This project answers two questions:

1. Which customer groups are most valuable, and which need attention? (segmentation)
2. Which customers are likely to stop buying? (churn prediction)

## Results so far

| Item | Result |
|---|---|
| Rows kept after cleaning | 1,025,292 of 1,067,371 (3.94% removed, limit is 5%) |
| Customers with a real ID, after RFM filtering | 5,832 |
| Customer segments | 4 (K-Means, k=4) |
| Churn dataset | 5,256 customers, 56.45% churned |
| Best churn model | Random Forest, test ROC-AUC **0.794** (target: 0.75) |

### Segments

| Segment | Customers | Median recency (days) | Median orders | Recommended action |
|---|---|---|---|---|
| Champions / VIP | 1,189 | 17 | 13 | Reward with early access and loyalty perks |
| Promising / Active | 1,253 | 24 | 3 | Nurture with recommendations and bundles |
| At-Risk High-Value | 1,440 | 184 | 4 | Win-back campaigns before they fully churn |
| Dormant / Lost | 1,950 | 402 | 1 | Low-cost automated re-engagement only |

K-Means with k=4 was chosen to meet the business need for at least 4 actionable segments. Silhouette score is highest at k=2 (0.438) and 0.367 at k=4. See `reports/figures/kmeans_elbow_silhouette.png`.

### Churn model

| Model | Precision | Recall | F1 | Test ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.761 | 0.726 | 0.743 | 0.791 | 0.791 |
| Random Forest | 0.756 | 0.731 | 0.743 | 0.794 | 0.800 |

Full report: `reports/churn_model_metrics.md`.

**How churn is defined (no data leakage):**
- Cutoff date: 90 days before the last transaction in the data (10 Sep 2011).
- Features use only purchases **before** the cutoff.
- A customer is "churned" if they made **no purchase in the 90 days after** the cutoff.
- The 90-day window covers the Sep to Dec holiday peak. A 30 or 60 day window is a planned comparison.

## Data decisions

- **Guest transactions:** about 22% of rows have no Customer ID. They are kept (flagged `Is_Registered = False`) so revenue and sales analysis stays complete. They are excluded from RFM, clustering and churn, because they cannot be tied to a single customer.
- **Cleaning:** 34,335 exact duplicates and 7,742 non-product or test rows (postage, bank charges, adjustments) are removed. Cancellations are flagged. Extreme outliers are capped.
- **Column meanings:** see `reports/data_dictionary.md`.

## Project structure

```
data/
  raw/          original CSV (never modified)
  interim/      cleaned transactions
  processed/    RFM, segments and churn features
notebooks/      numbered analysis notebooks (1.0 EDA, 2.0 RFM, 3.0 segmentation, 4.0 churn)
src/
  dataset.py            cleaning pipeline
  features.py           RFM feature engineering
  modeling/
    segmentation.py     K-Means customer segmentation
    churn_data.py       leakage-free churn dataset
    train.py            churn model training
models/         saved models (.pkl)
reports/        metrics, data dictionary, figures/
tests/          unit tests
```

## How to run

Python 3.12 or newer is recommended.

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Run from the repo root, in this order:

```bash
python src/dataset.py                 # raw -> data/interim
python src/features.py                # RFM features
python src/modeling/segmentation.py   # customer segments + K-Means model
python src/modeling/churn_data.py     # churn features and labels
python src/modeling/train.py          # trains models, saves best to models/
python -m pytest tests                # tests
```

## Status

| Part | Status |
|---|---|
| Data cleaning and EDA | Done |
| RFM and K-Means segmentation | Done |
| Churn dataset and baseline models | Done |
| Model tuning, ROC and confusion matrix plots (`4.0` notebook) | In progress |
| PostgreSQL load and SQL extraction | Planned |
| `predict.py` (inference script) | Planned |
| BI dashboard (Power BI or Tableau) | Planned |
| Final business report in `reports/` | Planned |
