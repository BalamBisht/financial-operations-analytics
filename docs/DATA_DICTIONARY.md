# Data Dictionary

All data in this project is **synthetically generated** (see `src/financial_analytics.py`,
Section 1) with a fixed random seed (`42`), so re-running the pipeline reproduces
these files byte-for-byte. This was a deliberate design choice for a teaching
portfolio project: it allows the full pipeline to be shared and reproduced
without exposing any real, confidential business data.

## `data/raw/financial_customers.csv`
One row per customer (5,000 rows).

| Column | Type | Description |
|---|---|---|
| customer_id | string | Unique customer identifier (`CUST_000001` ...) |
| signup_date | date | Date the customer signed up |
| segment | category | `Enterprise`, `Mid-Market`, `Small Business`, `Startup` |
| industry | category | Customer's industry vertical |
| country | category | Customer's country |
| plan | category | Subscription plan: `Basic`, `Professional`, `Business`, `Enterprise` |
| mrr | float | Monthly Recurring Revenue for the customer's plan ($) |
| contract_length | int | Contract length in months |
| number_of_users | int | Number of seats/users on the account |
| support_tickets | int | Total support tickets raised |
| usage_score | int (0-100) | Product usage/engagement score |
| nps_score | int (-100-100) | Net Promoter Score |
| churn_date | date (nullable) | Date the customer churned, if applicable |
| is_churned | 0/1 | Whether the customer has churned |
| lifetime_months | float | Customer lifetime in months |
| total_revenue | float | Total revenue billed to date ($) |
| clv | float | Customer Lifetime Value ($) |
| last_transaction_date | date | Date of the customer's most recent transaction |
| recency_days | int | Days since the last transaction |
| transaction_count | int | Total number of transactions |
| avg_transaction_value | float | Average transaction amount ($) |
| cohort_month | string (YYYY-MM) | Signup cohort, used for cohort analysis |

## `data/raw/financial_transactions.csv`
One row per transaction (~135,000 rows).

| Column | Type | Description |
|---|---|---|
| transaction_id | string | Unique transaction identifier |
| customer_id | string | Foreign key to `financial_customers.csv` |
| transaction_date | date | Date of the transaction |
| amount | float | Transaction amount ($) |
| transaction_type | category | `New`, `Subscription`, `Upgrade`, etc. |
| status | category | `Completed`, `Failed`, `Refunded` |
| payment_method | category | `Credit Card`, `ACH`, `Wire Transfer`, `PayPal` |
| year_month | string | Calendar year-month of the transaction |
| cohort_month | string | Customer's signup cohort month |
| transaction_month | string | Month index used for cohort alignment |

## `data/raw/monthly_revenue.csv`
Monthly revenue aggregated across all customers (60 monthly rows, 2020-2024).

| Column | Type | Description |
|---|---|---|
| year_month | date | First day of the month |
| total_revenue | float | Total revenue for the month ($) |
| num_transactions | int | Number of transactions in the month |
| avg_transaction | float | Average transaction value ($) |
| unique_customers | int | Distinct paying customers in the month |
| revenue_growth | float | Month-over-month revenue growth (%) |
| month | int | Calendar month (1-12) |
| quarter | int | Calendar quarter (1-4) |
| is_q4 | 0/1 | Flag for Q4 seasonality |

## `data/processed/at_risk_customers.csv`
Output of the churn model: customers with predicted churn probability > 50%.

| Column | Type | Description |
|---|---|---|
| customer_id | string | Customer identifier |
| segment / industry / plan | category | Customer attributes |
| mrr | float | Monthly Recurring Revenue at risk |
| clv | float | Customer Lifetime Value |
| usage_score / nps_score / support_tickets | numeric | Behavioral signals |
| churn_probability | float (0-1) | Model-predicted probability of churn |
| risk_category | category | `Medium Risk`, `High Risk`, `Very High Risk` |

## `data/processed/rfm_segmentation.csv`
RFM (Recency, Frequency, Monetary) segmentation of active customers.

| Column | Type | Description |
|---|---|---|
| customer_id | string | Customer identifier |
| segment / plan | category | Customer attributes |
| recency | int | Days since last transaction |
| frequency | int | Number of transactions |
| monetary | float | Total spend ($) |
| r_score / f_score / m_score | int (1-5) | RFM quintile scores |
| customer_segment | category | Named RFM segment (e.g. `Champions`, `Promising`, `At Risk`) |
