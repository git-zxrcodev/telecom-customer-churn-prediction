# Telecom Customer Churn Prediction

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)

---

## Overview
This project demonstrates an **end-to-end machine learning pipeline** for predicting customer churn, including:

- Data downloading & cleaning  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Model training & evaluation  
- Actionable retention strategy

The goal is to identify high-risk churn customers and provide stakeholders with prioritized recommendations to reduce churn and optimize marketing spend.

---

## Problem
Customer churn directly reduces revenue and increases acquisition costs. Predicting churn early — and understanding why it happens — allows the business to:

- Prioritize retention spend efficiently  
- Target high-risk cohorts with actionable interventions  
- Maximize ROI on marketing and customer success initiatives  

---

## Dataset
The dataset is **not included in this repository** due to size and licensing restrictions.

- **Source:** [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **License:** Data files © Original Authors  

> To download the dataset, run the `data_downloading.ipynb` script included in this repo.

---

## Key Findings
- **High early-tenure churn:** Most churn occurs in months 1–5.  
- **Payment friction:** Customers using manual payment methods (checks) have higher churn.  
- **Service mismatch:** Fiber customers without proper technical support show elevated churn risk.  

---

## Detailed Conclusion
- **Located:** docs/Conclusion.md

---

## Modeling Summary
| Model                | Strength                              | Use Case                                     |
|---------------------|--------------------------------------|--------------------------------------------|
| Logistic Regression  | High recall / aggressive             | Catch most churners                         |
| XGBoost              | Sensitive to non-linear interactions | Robust structure, good balance of metrics  |
| LightGBM             | Balanced                             | Trade-off between sensitivity & false positives |
| Random Forest        | Conservative                         | Minimize wasted marketing spend             |

---

## Usage

1. **Clone the repository**
```bash
git clone https://github.com/git-zxrcodev/telecom-customer-churn-prediction
cd telecom-churn-prediction

