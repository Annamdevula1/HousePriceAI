# 🏠 HOUSEPRICEAI
## Intelligent House Price Prediction & Analytics Dashboard

> **Data Analytics + AI/ML Portfolio Project**  
> *Turn raw house-price data into actionable insights and ML-powered price estimates.*

---

## Overview

**HOUSEPRICEAI** is a complete end-to-end house price analytics and prediction application built with Python, Streamlit, Pandas, Scikit-learn, and Plotly. The application transforms raw property data into an interactive analytics dashboard with machine learning–based price estimation.

---

## Problem Statement

House prices vary depending on property characteristics such as size, location, condition, number of bedrooms, and other attributes. Raw house-price data alone does not provide clear decision-making information. This project bridges that gap by:

1. Cleaning and structuring raw property data
2. Exploring price patterns through interactive visualisations
3. Training and comparing multiple ML regression models
4. Providing a live price-estimation interface
5. Generating dynamic, data-driven insights

---

## Objectives

- Understand house price distributions and patterns
- Identify features most associated with price variation
- Compare prices across locations and conditions
- Build and evaluate regression ML models (Linear Regression, Random Forest, Gradient Boosting)
- Provide an interactive house price prediction form
- Generate dynamic insights backed by actual dataset statistics

---

## Dataset

| Property | Value |
|---|---|
| File | `House Price Prediction Dataset.csv` |
| Rows | 2,000 |
| Columns | 10 |
| Features | Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage |
| Target | Price |
| Missing values | 0 |
| Duplicate rows | 0 |

**Categorical columns:** Location (Downtown, Suburban, Urban, Rural), Condition (Excellent, Good, Fair, Poor), Garage (Yes, No)

### Dataset Link

Link: https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset

---

## Features

| Dashboard Section | Description |
|---|---|
| 🏠 Dashboard | KPI cards, overview charts, quick insights |
| 📊 Analytics | Filtered interactive price exploration |
| 🔍 Data Explorer | Schema, quality report, statistics |
| 🤖 Prediction | ML price estimation form with result card |
| 📈 Model Performance | Metrics table, comparison chart, residuals, feature importance |
| 💡 Insights | Dynamic data-driven observations |
| ℹ️ About | Project info, Responsible AI, limitations |

---

## Technology Stack

| Category | Libraries |
|---|---|
| Frontend | Streamlit, HTML, CSS |
| Data Analytics | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Plotly |
| Model Persistence | Joblib |
| Language | Python 3.11+ |

---

## Architecture

```
User
  │
  ▼
Streamlit Frontend (app.py)
  │
  ├── src/data_processing.py   ← Load, validate, clean CSV
  ├── src/analytics.py         ← KPIs, summaries, filters
  ├── src/eda.py               ← Plotly visualisation functions
  ├── src/feature_engineering.py ← Derived features, ColumnTransformer pipeline
  ├── src/ml_model.py          ← Train / evaluate / persist models
  ├── src/prediction.py        ← Apply model to new property input
  └── src/insights.py          ← Dynamic plain-English insights
  │
  ├── models/
  │   ├── house_price_model.pkl   ← Saved best model pipeline
  │   └── model_results.pkl       ← Saved evaluation results
  │
  └── House Price Prediction Dataset.csv
```

---

## Data Preprocessing

1. **Load** the existing CSV from the project root
2. **Validate** schema (expected columns present, sufficient rows)
3. **Clean**: remove duplicates, drop invalid prices/areas, clamp bedroom/floor outliers, standardise categorical casing
4. **Feature Engineering**: add `TotalRooms`, `PropertyAge`, `AreaPerRoom`
5. **ColumnTransformer**: numerical → median impute + StandardScaler; categorical → mode impute + OneHotEncoder
6. **Train/test split**: 80/20

---

## Exploratory Data Analysis

Charts generated dynamically from actual dataset columns:

- Price distribution (histogram + box)
- Price vs Area scatter with trendline
- Price by Location (box plots, bar chart)
- Price by Condition (bar chart)
- Price by Bedrooms (line chart)
- Price by Floors and Garage (bar charts)
- Price by Build Decade (bar chart)
- Correlation heatmap
- Violin plots (Condition × Location)
- Outlier overview (box plots)

---

## Machine Learning

Three regression models are trained and compared:

| Model | Notes |
|---|---|
| Linear Regression | Baseline; fast; interpretable coefficients |
| Random Forest Regressor | Ensemble of 200 trees; handles non-linearity |
| Gradient Boosting Regressor | Sequential boosting; often strongest on tabular data |

**Model selection:** Automatically selected based on highest R² on the held-out test set.

---

## Model Evaluation

Metrics reported from actual test-set evaluation (no fabrication):

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error — avg absolute prediction error (same units as price) |
| RMSE | Root Mean Squared Error — penalises large errors more |
| R² | Coefficient of Determination — proportion of price variance explained (0–1) |

The application shows:
- Comparison table and bar charts for all three models
- Actual vs Predicted scatter plot
- Residual distribution and residuals vs predicted
- Feature importance (top 15 features)

---

## Prediction Workflow

```
User fills in property form
  → validate_inputs()
  → build_input_df()         ← adds derived features
  → pipeline.predict()       ← same preprocessing as training
  → display result card
```

---

## Project Structure

```
.
├── app.py                              ← Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
│
├── House Price Prediction Dataset.csv  ← Original dataset (do not move)
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── ml_model.py
│   ├── prediction.py
│   ├── analytics.py
│   └── insights.py
│
└── models/
    ├── house_price_model.pkl           ← Generated on first run
    └── model_results.pkl               ← Generated on first run
```

---

## Installation

```bash
# 1. Clone or download the project
cd "IBM SkillsBuild Data Analytics Project"

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your default browser.

**First run:** Models are trained automatically (takes ~10–30 seconds). Subsequent runs load persisted models instantly.

---

## Example Usage

1. Open the app in your browser.
2. Navigate to **🏠 Dashboard** for an overview of KPIs and price patterns.
3. Go to **📊 Analytics** to filter by location, condition, garage, bedrooms, and price range.
4. Visit **🔍 Data Explorer** to inspect the raw dataset quality.
5. Use **🤖 Prediction** to estimate the price of a custom property.
6. Check **📈 Model Performance** for transparent model evaluation.
7. Read **💡 Insights** for data-driven observations.

---

## Responsible AI

- Predictions are statistical estimates, not formal appraisals.
- Results depend on the quality and representativeness of historical training data.
- Historical patterns do not guarantee future property prices.
- Predictions should not be the sole basis for financial or property decisions.
- Feature importance reflects association with model predictions — not causation.
- Risk of bias exists if training data is incomplete or unrepresentative.
- No sensitive personal attributes are used in this model.

---

## Limitations

- Model accuracy is bounded by the features available in this dataset.
- Factors such as neighbourhood amenities, school ratings, or transport proximity are not included.
- The model is trained on a static snapshot; real-world prices change over time.
- Predictions outside the training data distribution (extrapolation) may be less reliable.

---

*HOUSEPRICEAI — Data Analytics + AI/ML Portfolio Project*
