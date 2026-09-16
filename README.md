# Financial Operations Analytics
### Revenue Forecasting · Customer Churn Prediction · Profitability & Cohort Analysis

An end-to-end data analytics project for a SaaS/subscription-style business, covering
three connected business problems: **predicting future revenue**, **finding customers
who are about to churn before they leave**, and **understanding which customers/segments
are actually profitable**.

---

## 1. Business Problem

A subscription business needs answers to three questions every finance/growth team asks:

1. **How much revenue will we make next month/quarter?** (planning & budgeting)
2. **Which customers are about to cancel, and how much revenue is at risk?** (retention)
3. **Which customers/segments/plans are actually worth the most, long-term?** (where to invest)

This project builds a repeatable analytics pipeline that answers all three from a single
transaction dataset.

## 2. Dataset

The dataset is **synthetically generated** (fixed random seed, `numpy.random.seed(42)`)
to mimic a realistic 5-year (2020–2024) SaaS subscription business: 5,000 customers,
~135,000 transactions, 4 subscription plans, 4 customer segments, 8 industries, and 7
countries. Using generated data means the full project — code, data, and results — can be
shared publicly without exposing any real company's financials.

Full column-level definitions are in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

| File | Rows | Description |
|---|---|---|
| `data/raw/financial_customers.csv` | 5,000 | Customer master data (plan, MRR, churn status, CLV, usage, NPS...) |
| `data/raw/financial_transactions.csv` | ~135,000 | Full transaction history |
| `data/raw/monthly_revenue.csv` | 60 | Monthly revenue aggregates, 2020–2024 |
| `data/processed/at_risk_customers.csv` | 652 | Customers flagged as high churn risk by the model |
| `data/processed/rfm_segmentation.csv` | 4,345 | RFM (Recency/Frequency/Monetary) segments for active customers |

## 3. Tools & Techniques

**Language/stack:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Statsmodels,
(optionally) Prophet.

| Business Question | Technique |
|---|---|
| Revenue forecasting | Time-series decomposition, ADF stationarity test, ACF/PACF analysis, **ARIMA**, (optional) **Prophet** |
| Churn prediction | **Logistic Regression**, **Random Forest**, **Gradient Boosting**, ROC/AUC evaluation, feature importance |
| Customer value | **RFM segmentation**, **Customer Lifetime Value (CLV)**, cohort retention analysis |
| Customer grouping | **K-Means clustering** on behavioral/financial features |
| Risk management | Probability-based risk stratification (Low / Medium / High / Very High risk) |

## 4. Methodology (what the pipeline actually does)

1. **Data generation** — builds the synthetic customer/transaction/revenue tables with
   realistic seasonality, growth, and churn patterns.
2. **Exploratory analysis** — customer distribution by segment, MRR by plan, churn rate
   by segment, CLV distribution, usage patterns (`visualizations/01_initial_exploration.png`).
3. **Revenue forecasting** — decomposes the revenue series into trend/seasonal/residual
   components, checks stationarity, fits an ARIMA model, and forecasts the next 12 months
   with a 95% confidence interval (`visualizations/02-06`).
4. **Churn prediction** — engineers behavioral features (recency, usage, tickets, tenure),
   trains and compares three classifiers, evaluates them with ROC/AUC and a confusion
   matrix, and extracts feature importance (`visualizations/07-10`).
5. **Cohort & RFM analysis** — tracks retention by signup cohort and segments the active
   base by Recency/Frequency/Monetary value (`visualizations/11-14`).
6. **Profitability analysis** — rolls everything up by segment/plan into a profitability
   view and a single executive dashboard (`visualizations/15-16`).
7. **Reporting** — writes `outputs/kpi_summary.txt` and `outputs/EXECUTIVE_SUMMARY_FINANCIAL.txt`
   directly from the computed results (no manual/typed numbers).

## 5. Key Results

*(These are the actual, current results of re-running the pipeline in this repo — see
`outputs/kpi_summary.txt` for the full report.)*

**Revenue**
- Total historical revenue: **$35,834,169.55** · Current MRR: **$1,160,806** · ARR: **$13,929,672**
- 12-month ARIMA revenue forecast: **$15,715,279.91** (MAPE **0.73%**)
- 6-month revenue growth rate: **+13.26%**

**Churn**
- Overall churn rate: **13.12%** (656 of 5,000 customers)
- **652 customers** flagged as at-risk (>50% predicted churn probability), representing
  **$166,748** in MRR / **~$2.0M** in potential annual revenue loss
- Random Forest / Gradient Boosting achieved near-perfect separation (ROC AUC ≈ 1.0) on
  this dataset; **recency, customer lifetime, and transaction count** were the strongest
  churn predictors (`visualizations/09_churn_feature_importance.png`)

**Customer Value**
- Average CLV: **$7,166.83** · Median CLV: **$2,907.19**
- Assumed CAC of $500 → **CLV:CAC ratio of 14.33x**, average payback period **5.0 months**

> Note on model performance: the near-perfect churn AUC reflects that this is a
> synthetic dataset with clean, strongly-correlated churn signals built in by design
> — it demonstrates the modeling pipeline correctly, but shouldn't be read as a
> real-world benchmark.

## 6. Visualizations

All charts are in [`visualizations/`](visualizations/) and are regenerated by
`src/financial_analytics.py`:

| # | File | What it shows |
|---|---|---|
| 01 | `01_initial_exploration.png` | Customer segments, MRR by plan, churn by segment, revenue trend, CLV, usage |
| 02 | `02_ts_decomposition.png` | Revenue trend / seasonal / residual decomposition |
| 03 | `03_acf_pacf_analysis.png` | ACF/PACF for original & differenced revenue series |
| 04 | `04_arima_forecast.png` | ARIMA 12-month forecast with confidence interval |
| 07 | `07_churn_analysis.png` | Churn rate by segment, contract length, usage, NPS, support tickets, industry |
| 08 | `08_churn_model_evaluation.png` | Model metrics, ROC curves, confusion matrix, churn probability distribution |
| 09 | `09_churn_feature_importance.png` | Top churn predictors (Random Forest) |
| 10 | `10_risk_stratification.png` | Customer & MRR distribution by risk category |
| 11 | `11_cohort_retention.png` | Retention curves by signup cohort |
| 12 | `12_revenue_cohorts.png` | Revenue by cohort over time |
| 13 | `13_rfm_analysis.png` | RFM segment sizes and characteristics |
| 14 | `14_clv_analysis.png` | CLV distribution and drivers |
| 15 | `15_profitability_dashboard.png` | Profitability by segment/plan |
| 16 | `16_FINAL_EXECUTIVE_DASHBOARD.png` | One-page executive summary dashboard |

> `05_prophet_forecast.png` and `06_prophet_components.png` are only generated when the
> optional `prophet` package is installed — see [Known Limitations](#9-known-limitations--what-couldnt-be-verified).

## 7. Project Structure

```
financial-operations-analytics/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/                          # Generated source data
│   │   ├── financial_customers.csv
│   │   ├── financial_transactions.csv
│   │   └── monthly_revenue.csv
│   └── processed/                    # Model/analysis outputs
│       ├── at_risk_customers.csv
│       └── rfm_segmentation.csv
│
├── notebooks/
│   └── Financial_Operations_Analytics.ipynb   # Full analysis, notebook form
│
├── src/
│   └── financial_analytics.py        # Same analysis as a runnable script
│
├── outputs/
│   ├── kpi_summary.txt
│   └── EXECUTIVE_SUMMARY_FINANCIAL.txt
│
├── visualizations/                   # 14 PNG charts (see table above)
│
└── docs/
    └── DATA_DICTIONARY.md
```

## 8. How to Run

```bash
# 1. Clone this repository, then move into it
git clone <this-repository-url>
cd financial-operations-analytics

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (regenerates data, models, CSVs, and all charts)
python src/financial_analytics.py
```

Or open and run [`notebooks/Financial_Operations_Analytics.ipynb`](notebooks/Financial_Operations_Analytics.ipynb)
in Jupyter — run all cells from the `notebooks/` folder (or update the paths if you move it).

**Runtime:** a few minutes on a standard laptop. The pipeline is deterministic
(`random_seed=42`), so re-running it reproduces the exact same data and metrics.

## 9. Known Limitations / What Couldn't Be Verified

Being transparent about the source material this project was built from:

- **Data is synthetic**, generated for teaching/portfolio purposes — not real company
  data. All KPIs/results above are real outputs of *this* pipeline on that synthetic
  data, not invented numbers.
- **Prophet forecasting is optional.** The original materials referenced
  `05_prophet_forecast.png` and `06_prophet_components.png`, but the `prophet` package
  isn't installed by default in this environment, so those two charts are only produced
  if you `pip install prophet` yourself. Everything else (ARIMA forecast, churn models,
  cohorts, RFM, CLV, profitability) runs with the base `requirements.txt` and needs no
  extra setup.
- **The original notebook never actually wrote `EXECUTIVE_SUMMARY_FINANCIAL.txt`** — it
  only printed a message saying it had. This has been fixed in `src/financial_analytics.py`
  / the notebook so the file is genuinely generated in `outputs/`.
- **The churn model's near-perfect accuracy (ROC AUC ≈ 1.0)** is a property of the
  synthetic data generation (churn is a near-deterministic function of a few features by
  design), not a claim about real-world churn model performance.

## 10. Author

**Balam Singh Bisht** — Data Analyst
