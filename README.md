# Retail-Ecommerce---Customer-Segmentation-and-Churn-Prediction

**Project:** Retail-Ecommerce--CSCP  
**Author:** Aman Singh  
**Version:** 1.0.0  

Customer segmentation (RFM Analysis, K-Means clustering) and churn prediction for an e-commerce retail platform. Project initialized using the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template.

## Project Structure

```
Retail-Ecommerce--CSCP/
│
├── data/
│   ├── raw/                  <- Original, immutable raw data
│   ├── interim/              <- Intermediate transformed data
│   ├── processed/            <- Final canonical datasets for modeling
│   └── external/             <- Third-party data
│
├── notebooks/
│   ├── 1.0-eda.ipynb                    <- Exploratory Data Analysis
│   ├── 2.0-rfm-analysis.ipynb           <- RFM metrics & customer scoring
│   ├── 3.0-customer-segmentation.ipynb  <- Clustering & customer segmentation
│   └── 4.0-churn-prediction.ipynb       <- Churn prediction classification models
│
├── src/
│   ├── __init__.py           <- Makes src a Python module
│   ├── dataset.py            <- Scripts to download or generate data
│   ├── features.py           <- Scripts to turn raw data into features for modeling
│   └── modeling/
│       ├── __init__.py
│       ├── train.py          <- Scripts to train models
│       └── predict.py        <- Scripts to use trained models to make predictions
│
├── models/
│   └── churn_model.pkl       <- Trained and serialized model
│
├── reports/
│   └── figures/              <- Generated graphics and figures for reporting
│
├── dashboard/
│   └── retail_dashboard.pbix <- Interactive Power BI / business dashboard
│
├── docs/                     <- Sphinx documentation project
├── references/               <- Data dictionaries, manuals, and explanatory materials
├── requirements.txt          <- Dependencies file for reproducing the environment
├── setup.py                  <- Makes project pip installable (pip install -e .)
├── Makefile                  <- Makefile with commands like `make data` or `make train`
├── README.md                 <- Top-level project documentation
└── .gitignore                <- Git ignore rules
```

## Setup & Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
