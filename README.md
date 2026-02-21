# Telecom Customer Churn Prediction

An end-to-end machine learning project that identifies customers at risk of churning for a telecommunications company. The project covers data ingestion, exploratory data analysis, feature engineering, and training four classifiers — with actionable retention recommendations grounded in the data.

---

## Table of Contents

- [Telecom Customer Churn Prediction](#telecom-customer-churn-prediction)
  - [Table of Contents](#table-of-contents)
  - [1. Business Problem \& Objective](#1-business-problem--objective)
  - [2. Dataset](#2-dataset)
    - [Feature Glossary](#feature-glossary)
  - [3. Project Structure](#3-project-structure)
  - [4. Tech Stack](#4-tech-stack)
  - [5. Notebooks Overview](#5-notebooks-overview)
  - [6. Key EDA Insights](#6-key-eda-insights)
    - [Top Churn Drivers (by Mutual Information \& Correlation)](#top-churn-drivers-by-mutual-information--correlation)
    - [Loyalty Signal](#loyalty-signal)
    - [Negligible Factors](#negligible-factors)
  - [7. Feature Engineering](#7-feature-engineering)
  - [8. Modelling Results](#8-modelling-results)
  - [9. Retention Plan](#9-retention-plan)
  - [10. Run This Project Locally](#10-run-this-project-locally)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Run the Notebooks](#run-the-notebooks)
    - [Deactivate the Virtual Environment](#deactivate-the-virtual-environment)
  - [11. Contact](#11-contact)

---

## 1. Business Problem & Objective

Acquiring a new customer costs significantly more than retaining an existing one. For a telecom provider with a **~26% churn rate**, understanding *why* customers leave — and acting before they do — is a direct revenue lever.

**Objective:** Analyse customer records to uncover the main drivers of churn, engineer predictive features, and train classification models that can score customers by churn risk so the business can target the right people with the right retention offer at the right time.

---

## 2. Dataset

- **Source** | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records** | 7,043 customers
- **Features** | 21 (demographics, services subscribed, contract & billing)
- **Target** | `Churn` — Yes / No
- **Class balance** | ~73.5% No churn / ~26.5% Churn

### Feature Glossary

| Column | Description |
|---|---|
| `customerID` | Unique customer identifier |
| `gender` | Male / Female |
| `SeniorCitizen` | Senior citizen status (0 = No, 1 = Yes) |
| `Partner` | Has a partner (Yes / No) |
| `Dependents` | Has dependents (Yes / No) |
| `tenure` | Months with the company |
| `PhoneService` | Phone service (Yes / No) |
| `MultipleLines` | Multiple phone lines |
| `InternetService` | DSL / Fiber optic / No |
| `OnlineSecurity` | Online security add-on |
| `OnlineBackup` | Online backup add-on |
| `DeviceProtection` | Device protection add-on |
| `TechSupport` | Tech support add-on |
| `StreamingTV` | TV streaming |
| `StreamingMovies` | Movie streaming |
| `Contract` | Month-to-month / One year / Two year |
| `PaperlessBilling` | Paperless billing (Yes / No) |
| `PaymentMethod` | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Cumulative charges |
| `Churn` | **Target** — Yes / No |

---

## 3. Project Structure

```
telecom-customer-churn-prediction/
│
├── notebooks/
│   ├── 01_ingestion_and_eda.ipynb        # Data ingestion, wrangling & EDA
│   ├── 02_feature_engineering.ipynb      # Chi-squared tests & feature creation
│   ├── 03_predictive_modeling.ipynb      # Model training, evaluation & inference
│   └── 04_transform_for_dashboard.ipynb  # Label mapping & dashboard CSV export
│
├── src/
│   ├── clean_column_names.py    # Standardises column names to snake_case
│   ├── df_overview.py           # Quick dataset summary utility
│   ├── download_data.py         # Kaggle download helper (kagglehub)
│   ├── feature_engineering.py   # Reusable feature engineering function
│   └── schemas.py               # Schema contracts for each pipeline stage
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 4. Tech Stack

| Category | Libraries |
|---|---|
| **Data manipulation** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn, missingno |
| **Machine learning** | scikit-learn, imbalanced-learn |
| **Gradient boosting** | XGBoost, LightGBM |
| **Model persistence** | joblib |
| **Data acquisition** | kagglehub |
| **Runtime** | Python 3.12, Jupyter |

---

## 5. Notebooks Overview

- `01_ingestion_and_eda.ipynb` | Downloads data via `kagglehub`, validates schema, wrangles types, performs full univariate & multivariate EDA (correlation, mutual information, heatmaps), saves processed CSV
- `02_feature_engineering.ipynb` | Runs chi-squared tests on all categorical features, creates 5 engineered features, drops statistically non-significant and redundant columns, saves featured CSV
- `03_predictive_modeling.ipynb` | Trains four classifiers with cross-validated hyperparameter search and imbalance handling, evaluates on held-out test set, saves model artefacts + JSON metadata, demonstrates inference on a new customer
- `04_transform_for_dashboard.ipynb` | Translates model-ready encodings (ordinal ints, booleans) into human-readable category labels and exports a dashboard-ready CSV to `data/04_dashboard/`

---

## 6. Key EDA Insights

Overall churn rate is **~26%**, with a heavily imbalanced dataset (~5,163 retained vs ~1,869 churned).

### Top Churn Drivers (by Mutual Information & Correlation)

- **Month-to-month contract & early tenure**: 50%+ churn rate in the first 5 months of tenure — the single strongest driver & the critical onboarding window

![churn rate by tenure and contract](images/churn%20rate%20by%20tenure%20and%20contract%20type.png)

- **No online security**: ~20.8% weighted churn; marker of low service engagement
- **No tech support**: ~20.6% weighted churn; especially acute among fiber optic users
- **Fiber optic internet**: ~18.4% weighted churn; price sensitivity or service-quality gap vs. competitors
- **No partner / no dependents**: 17–22% weighted churn; less "anchored" customers
- **Electronic check payment**: ~15.2% weighted churn; friction or payment-failure risk

![churn volume by tenure and payment method](images/churn%20volume%20by%20tenure%20and%20payment%20method.png)

- **High monthly charges ($50–$110)**: Positive correlation with churn

![monthly charges distribution by churn](images/monthly%20charges%20distribution%20by%20churn.png)

### Loyalty Signal

- **Tenure**: the strongest negative predictor of churn — the longer a customer stays, the less likely they are to leave. Customers on two-year contracts show dramatically lower churn (~2–3%) compared to month-to-month (~43%).

### Negligible Factors

- **Gender**: virtually no churn difference (Female 26.9% vs Male 26.2%); excluded from modelling.
- **Phone service**: limited standalone predictive lift; excluded from modelling.

---

## 7. Feature Engineering

Five features were engineered from the raw columns after validating statistical significance with chi-squared tests:

| Feature | Type | Logic |
|---|---|---|
| `contract_stability` | Ordinal (1–3) | Month-to-month = 1, One year = 2, Two year = 3 |
| `high_risk_tenure` | Ordinal | Tenure bins: 1–4 months = high risk, 5–12 = medium risk, 13+ = low risk |
| `fiber_no_support` | Boolean | `internet_service == 'Fiber optic'` AND `tech_support == 'No'` |
| `manual_payment_early` | Boolean | Manual payment method (electronic/mailed check) AND `tenure <= 6` |
| `high_risk_new_monthly` | Boolean | `tenure <= 6` AND `contract == 'Month-to-month'` |

**Columns dropped:**

| Column | Reason |
|---|---|
| `gender` | Not statistically significant (chi-squared p = 0.49) |
| `phone_service` | Not statistically significant (chi-squared p = 0.35) |
| `contract` | Replaced by `contract_stability` (ordinal) and `high_risk_new_monthly` |
| `total_charges` | Strong multicollinearity with `tenure`; `monthly_charges` is more informative |

---

## 8. Modelling Results

**Train / test split:** 80 / 20 stratified (5,625 train, 1,407 test). All models tuned with 5-fold stratified cross-validation optimising ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Imbalance Strategy |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Logistic Regression** | 0.7534 | 0.5242 | **0.7834** | **0.6281** | **0.8443** | SMOTE inside ImbPipeline |
| **XGBoost**  | 0.7427 | 0.5103 | 0.7968 | 0.6221 | 0.8428 | `scale_pos_weight` |
| LightGBM | 0.7406 | 0.5079 | 0.7701 | 0.6121 | 0.8347 | `class_weight='balanced'` |
| Random Forest | **0.7974** | **0.6679** | 0.4733 | 0.5540 | 0.8330 | None (conservative baseline) |

**Recommended model: XGBoost** — highest recall among tree-based models (0.797), ROC-AUC of 0.843, simple imbalance handling via `scale_pos_weight`, and fast retraining. Logistic Regression is the preferred fallback where interpretability is required.

Random Forest's high accuracy (79.7%) is misleading — it misses **53% of churners**, making it unsuitable as a default production model for this task.

**Top features confirmed by XGBoost gain importance:** `contract_stability`, `high_risk_tenure`, `fiber_no_support`, `tenure`, `monthly_charges`.

Each model is persisted as a `.joblib` file alongside a JSON sidecar containing best hyperparameters, test metrics, feature list, train/test sizes, and training time — enabling full reproducibility without re-running grid search.

---

## 9. Retention Plan

- Proactive onboarding campaign (check-in calls at month 1 & 3, welcome bundle), targeting new month-to-month customers, tenure 0–5 months
- Time-limited incentive to migrate M2M → annual contract, aiming at high-risk M2M customers
- Bundle online security & tech support into mid-tier plans by default for customers without add-ons
- Proactive tech support outreach for new fiber optic customers, using fiber optic × no tech support cohort feature
- Incentivise switch to automatic payment ($5/month discount for auto-pay), transitioning electronic check users

---

## 10. Run This Project Locally

### Prerequisites

- Python 3.12
- `pip`
- `git`

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/git-zxrcodev/telecom-customer-churn-prediction
cd telecom-customer-churn-prediction

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies (includes the local src package via -e .)
pip install -r requirements.txt
```

### Run the Notebooks

Execute the notebooks **in order**:

```
notebooks/01_ingestion_and_eda.ipynb      # downloads & explores data
notebooks/02_feature_engineering.ipynb   # engineers features
notebooks/03_predictive_modeling.ipynb   # trains & evaluates models
notebooks/04_transform_for_dashboard.ipynb # prepares dashboard CSV
```

The raw data will be downloaded automatically on the first run of notebook 01 via `kagglehub`.

### Deactivate the Virtual Environment

```bash
deactivate
```

---

## 11. Contact

**Author:** Pavlo Popovych

- [LinkedIn](https://www.linkedin.com/in/pavlo-popovych/)
- [GitHub](https://github.com/git-zxrcodev)
- [Email](mailto:pavlo.v.popovych@outlook.com) 
