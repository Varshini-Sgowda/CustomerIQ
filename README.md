# CustomerIQ — Intelligent Customer Analytics & Decision-Support Platform

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Gradient%20Boosting%20Trees-orange.svg)](https://scikit-learn.org)
[![Explainability](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-purple.svg)](https://shap.readthedocs.io)
[![Anomaly Detection](https://img.shields.io/badge/Anomaly%20Detection-Isolation%20Forest-red.svg)](https://scikit-learn.org)
[![Framework](https://img.shields.io/badge/Web%20Engine-Flask%203.1-emerald.svg)](https://flask.palletsprojects.com)
[![Testing](https://img.shields.io/badge/Tests-27%20Passed%20(100%25)-brightgreen.svg)](#running-tests)
[![Status](https://img.shields.io/badge/Production-Verified-success.svg)](#startup-instructions)

---

## 📌 Executive Overview

**CustomerIQ** is an enterprise-grade customer intelligence and autonomous decision-support system. It transforms raw commercial transactions into explainable forward-looking machine learning predictions, real-time behavioral change detection, scenario-based digital twin simulations, and prioritized marketing interventions.

Unlike traditional descriptive dashboards that merely summarize past sales, CustomerIQ delivers **predictive, prescriptive, and explainable intelligence** without external black-box LLM dependencies or data hallucination risks.

---

## 🏛 Architecture & Core Capabilities

```
project1/
├── customeriq/                     # Core Analytical Engine
│   ├── data_loader.py              # Ingestion, schema validation & temporal cutoff splitting
│   ├── feature_pipeline.py         # Point-in-time leak-free RFM & momentum feature engineering
│   ├── ltv_model.py                # Gradient Boosting Regressor, SHAP Explainer & 80% CI bounds
│   ├── profiles.py                 # Customer 360 dossier with epistemic provenance tagging
│   ├── copilot.py                  # Deterministic natural-language analytics grammar engine
│   ├── anomaly_detector.py         # Recency drift, velocity collapse & Isolation Forest anomalies
│   ├── future_lab.py               # Customer digital twin simulation with constraint validation
│   ├── action_planner.py           # Marketing budget optimizer (knapsack) & uplift modeling
│   ├── insight_engine.py           # Automated 8-point insight discovery cards
│   ├── drift_monitor.py            # Population Stability Index (PSI) & slice diagnostics
│   └── report_generator.py         # Multi-page PDF, multi-tab Excel & CSV compiler
├── data/
│   ├── Superstore_Cleaned.csv      # Primary transactional dataset (9,994 records, 793 accounts)
│   └── models/
│       └── ltv_model.pkl           # Serialized gradient boosted regressor & metadata
├── static/
│   ├── css/style.css               # Responsive dark/light theme design system
│   └── js/app.js                   # Client-side router, reactive UI state & Chart.js rendering
├── templates/
│   └── index.html                  # Multi-workspace interactive dashboard template
├── tests/                          # Full unit test suite (27 automated tests)
├── run.py                          # Application entry point
├── requirements.txt                # Python package dependencies
└── README.md                       # Platform documentation & governance guide
```

---

## 🎯 The Eight Intelligence Workspaces

### 1. Ask CustomerIQ (Natural-Language Analytics Copilot)
- **Natural-Language Understanding:** Translates ordinary English business inquiries into deterministic analytical operations.
- **Controlled Analytical Grammar:** Eliminates LLM hallucination and arbitrary code execution by routing through verified Pandas computation pipelines.
- **Seven Core Operations Supported:**
  1. *Which customers have the highest predicted future LTV?*
  2. *Which high-value customers have not purchased recently?*
  3. *Which customer segments contribute the most historical revenue?*
  4. *What factors influence the LTV predictions?*
  5. *Which customer groups have declining purchase frequency?*
  6. *How does average order value vary across customer segments?*
  7. *What changes when the prediction horizon or business assumptions change?*
- **Audit Provenance:** Displays exact filters applied, metrics computed, mathematical formulas, and suggested follow-up questions.
- **Demographic Rejector:** Transparently explains when a question involves uncollected attributes (e.g., customer age, gender, credit score).

### 2. Customer 360 Dossier & Dual Comparison
- **Epistemological Integrity:** Distinguishes metrics with visual badges:
  - `[OBSERVED_FACT]`: Realized sales, historical order count, recency timestamps.
  - `[MODEL_ESTIMATE]`: Machine learning predicted forward 12M LTV and 80% confidence interval.
  - `[HEURISTIC_RULE]`: Multi-dimensional quintile RFM segmentation and rule-based explanations.
  - `[SIMULATION_HYPOTHETICAL]`: Digital twin scenario adjustments.
- **SHAP Feature Attribution:** Computes exact TreeExplainer contributions for each customer, decomposing baseline expectations into positive and negative drivers.
- **Side-by-Side Dual Comparison:** Select any two customer accounts to view side-by-side behavioral variance, LTV divergence, and margin gaps.

### 3. Early-Warning Radar & Behavioral Change Detection
- **Multi-Tiered Monitoring:** Tracks dormancy risk, order velocity drop (>50% drop), spending contraction (>60% collapse), and excessive promotional discount exposure.
- **Isolation Forest Multivariate Outliers:** Tree ensemble algorithm ($n=100$) detects non-linear behavioral anomalies across all 20 engineered features.
- **Actionable Alert Cards:** Provides severity (`CRITICAL`, `WARNING`, `OPPORTUNITY`, `INFO`), affected customer ID, empirical evidence, detection method, recommended investigation, and direct link to customer view.
- **Fair Audit Principle:** Single transactions are never treated as proof of churn, fraud, or dissatisfaction.

### 4. Customer Future Lab (Digital Twin)
- **Scenario Simulator:** Allows users to modify customer recency, frequency, average basket size, promotional discounts, and momentum.
- **Mathematical Consistency:** Enforces strict boundary constraints (recency cannot exceed tenure; frequency $\ge 1$; average order value $> 0$; discounts bounded between 0% and 80%).
- **Trained Model Inference:** Evaluates hypothetical behavioral states directly using the trained Gradient Boosting model.
- **Defensible Prediction Uncertainty:** Displays 80% empirical cross-validation prediction intervals ($[P_{10}, P_{90}]$ residual bounds).
- **Causal Disclaimer:** Explains clearly that modifying a model feature reflects observational association and does not guarantee real-world causal impact.

### 5. Opportunity Center (Marketing Action Planner)
- **Budget Optimization:** Allocates marketing capital across accounts using a greedy 0/1 knapsack algorithm under configurable budget limits ($500 to $50,000).
- **Four Strategic Priorities:**
  1. `PREVENT_CHURN_HIGH_LTV`: Prioritize high-LTV accounts slipping into dormancy.
  2. `MAXIMIZE_EXPECTED_ROI`: Target accounts with the highest elasticity per dollar spent.
  3. `WINBACK_DORMANT`: Reactivate lapsed customers.
  4. `NURTURE_POTENTIAL_LOYALISTS`: Secure 2nd-purchase conversions.
- **Uplift Modeling Heuristics:** Models heterogeneous treatment responsiveness based on customer discount affinity and recency decay.
- **Baseline Benchmark:** Benchmarks prioritized strategy against uniform blanket mass campaigns, demonstrating superior capital efficiency.

### 6. Automated Insight Feed
- **Self-Directed Analytical Engine:** Automatically scans portfolio for:
  - *Revenue Concentration:* Lorentz curve integration and Gini coefficient ($0.38$) showing top 10% customers generate $30.6\%$ of revenue.
  - *Segment Migration:* Identifies $16.5\%$ of customer base slipping into At-Risk status.
  - *Behavioral Velocity:* Flags 156 accounts exhibiting order frequency deceleration.
  - *Emerging VIPs:* Surfaces high-growth accounts acquired within 365 days.
  - *Promotional Margin Erosion:* Identifies 215 accounts with $>20\%$ average discounts producing negative unit economics.
  - *Catalog Mono-Dependency:* Detects 120 single-category accounts ripe for cross-sell.
  - *Data Quality Health:* Validates schema conformity across 9,994 transactions.
  - *Model Governance:* Reports cross-validated out-of-fold generalization.
- **8-Point Structured Cards:** Formats every insight with title, supporting metrics, population, baseline, method, hypothesis, action, and evidence qualification.

### 7. Model Health & Drift Monitor
- **Population Stability Index (PSI):**
  $$\text{PSI} = \sum \left( (\text{Actual} - \text{Expected}) \times \ln\left(\frac{\text{Actual}}{\text{Expected}}\right) \right)$$
  Tracks input distribution shift against baseline training reference across all 20 features:
  - $\text{PSI} < 0.10$: Stable (Green)
  - $0.10 \le \text{PSI} < 0.25$: Moderate Drift (Amber)
  - $\text{PSI} \ge 0.25$: Significant Shift / Recalibrate (Red)
- **Missingness Audit:** Checks for nulls or infinite values across features ($0.0\%$ missingness verified).
- **Segment Slices:** Reports slice-level performance and margin metrics across customer segments.
- **Health Governance Principle:** Explicitly distinguishes feature distribution drift from accuracy degradation, reminding stakeholders that true performance requires future ground-truth outcomes.

### 8. Executive Business Reports Generator
- **Publication-Grade PDF Report:** Generates multi-page executive review PDF using ReportLab Platypus with executive KPI cards, segment breakdowns, opportunity rankings, critical alerts, and governance disclaimers.
- **Multi-Tab Excel Workbook:** Generates a structured 5-sheet workbook (`Executive Summary`, `Customer Registry`, `Action Plan`, `Early Warning Alerts`, `Feature Drift PSI`) via `openpyxl`.
- **Enriched CSV Dataset:** Exports full customer feature registry with predicted LTVs and interval bounds.

---

## 📊 Dataset Schema & Data Requirements

CustomerIQ operates on transactional purchase records. The primary dataset is `data/Superstore_Cleaned.csv`:
- **Dimensions:** 9,994 order rows $\times$ 28 columns
- **Unique Customers:** 793 accounts
- **Date Span:** January 3, 2014 to December 30, 2017 (4 full fiscal years)
- **Core Required Fields:**
  - `Order ID`, `Order Date`, `Customer ID`, `Customer Name`, `Segment`
  - `Sales`, `Profit`, `Quantity`, `Discount`
  - `Category`, `Sub-Category`, `Region`

---

## ⚙️ Installation & Setup Instructions

### 1. Prerequisites
- Python 3.10+ (Verified on Python 3.14.3)
- Windows, macOS, or Linux

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python run.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Running Tests

The application includes a comprehensive test suite covering feature engineering, model training, SHAP explainability, query routing, early warning alerts, digital twin constraints, campaign optimization, insight generation, and PSI drift monitoring.

To run all unit tests:
```bash
python -m unittest discover -s tests -v
```

**Test Results:**
```
Ran 27 tests in 19.858s
OK (27 passed, 0 failures, 0 errors)
```

---

## 🔒 Limitations & Ethical Boundaries

1. **Absence of Personal Demographics:** The platform intentionally excludes demographic inferences (age, gender, ethnicity) not present in the transactional data.
2. **Correlation vs. Causation:** LTV predictions and digital twin simulations reflect observational associations. Causal lift must be verified through randomized holdout groups (A/B testing).
3. **Stochastic Uncertainty:** Customer purchase timing involves natural real-world variance. Individual customer outcomes are never represented as certain; all forecasts include defensible 80% confidence intervals.

