# Customer Churn — EDA & Predictive Modeling

## Focus: Exploratory data analysis, feature-driven churn modeling, retention strategy, stakeholder-ready outputs

# TL;DR

This repository contains an end-to-end analysis and predictive modeling pipelines that identifie customers at high risk of churn and provide actionable, prioritized retention recommendations. Key findings show acute early-tenure churn risk, clear payment-related friction, and service/technical-support weaknesses (notably among Fiber customers). Models were developed to balance sensitivity (finding churners) and precision (reducing wasted outreach), enabling stakeholders to select a risk strategy that matches marketing budget and appetite.

## Why this project

Customer churn directly reduces revenue and increases acquisition costs. Predicting churn early — and understanding why it happens — allows the business to prioritize retention spend where it has the highest ROI. This project demonstrates the analytics and product thinking required to convert data into prioritized retention actions and measurable impact.

## What I did

Performed EDA to surface the primary drivers of churn.

Built and compared multiple supervised models (Logistic Regression, XGBoost, LightGBM, Random Forest) to produce churn risk scores at the customer level.

Created a recommended, prioritized retention plan (first 90 days) directly tied to EDA insights.

Produced stakeholder-facing artifacts (charts, suggested executive PDF, and a plan for an ROI calculator / dashboard).

# Key EDA findings

High early-tenure churn (the “Danger Zone”)

Churn peaks in months 1–5 of tenure.

Among new customers (≤ 10 months) on month-to-month contracts, churn exceeds 50%.

Overall month-to-month churn: 56.89%.

Payment friction

Manual payment methods (Electronic Check, Mailed Check) have materially higher churn (34.74%) vs automated billing (16.00%).

Effect is strongest in months 1–2 — suggests onboarding / billing friction.

Service mismatch (Fiber vs DSL)

Long-tenure Fiber customers show elevated churn relative to DSL, indicating quality, pricing, or expectation gaps.

Lack of technical support

Customers without technical support churn at 41.65%, a signal of unresolved service issues — particularly relevant for Fiber.

# Retention plan (first 90 days) — prioritized

Proactive onboarding / Health Checks (months 0–2)
Quick technical and billing checks to reduce early attrition (target month-to-month subscribers and manual payers).

Short-term incentives (months 1–4)
Time-bound value offers for high-risk cohorts to improve early conversion to longer contracts.

Automated-billing incentive
One-time credit/discount to move manual payers to autopay — reduces billing friction.

Contract migration nudges
Loyalty offers emphasizing long-term value for month-to-month customers.

Technical support & infrastructure focus for Fiber
Prioritize support outreach and stability improvements for at-risk Fiber customers.

# Modeling summary & interpretation

Logistic Regression — most aggressive: captures nearly 80% of churners (high recall). Good when missing churners is costly and marketing budget can tolerate false positives.

XGBoost — second-most aggressive: strong sensitivity with more structure for non-linear feature interactions.

LightGBM — stable / balanced: trade-off between sensitivity and false positives.

Random Forest — conservative: fewer false positives, better when reducing wasted spend is the primary goal.