"""
housepriceai_complete.py
========================
HOUSEPRICEAI - Intelligent House Price Prediction & Analytics Dashboard
Complete single file containing ALL backend and frontend code.

Backend modules included:
  - data_processing  (load, validate, clean)
  - analytics        (KPIs, summaries, filters)
  - eda              (Plotly chart functions)
  - feature_engineering (derived features, preprocessing pipeline)
  - ml_model         (train, evaluate, compare, persist)
  - prediction       (predict price)
  - insights         (dynamic insights)

Frontend:
  - app.py           (Streamlit dashboard - 7 pages)

Run:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================================================================
#
#  BACKEND - data_processing.py
#
# ==============================================================================

CSV_PATH = "House Price Prediction Dataset.csv"
TARGET_COL = "Price"
ID_COL = "Id"
NUMERICAL_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
CATEGORICAL_FEATURES = ["Location", "Condition", "Garage"]
FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
    """Load the CSV and raise a descriptive error if it is missing."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{CSV_PATH}'. "
            "Ensure 'House Price Prediction Dataset.csv' is in the project root."
        )
    df = pd.read_csv(CSV_PATH)
    return df


def validate_schema(df: pd.DataFrame) -> dict:
    """Return a dict of validation warnings (empty = OK)."""
    warnings = {}
    required = FEATURE_COLS + [TARGET_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        warnings["missing_columns"] = missing_cols
    if df.shape[0] < 10:
        warnings["too_few_rows"] = df.shape[0]
    return warnings


def quality_report(df: pd.DataFrame) -> dict:
    """Return a comprehensive data-quality summary dict."""
    report = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numerical_cols": df.select_dtypes(include="number").columns.tolist(),
        "categorical_cols": df.select_dtypes(include="object").columns.tolist(),
        "unique_values": {c: df[c].nunique() for c in df.columns},
        "categorical_uniques": {
            c: df[c].unique().tolist()
            for c in df.select_dtypes(include="object").columns
        },
    }
    return report


def clean_data(df: pd.DataFrame) -> tuple:
    """Apply cleaning steps. Returns (cleaned_df, log_of_steps)."""
    log = []
    original_len = len(df)

    df = df.drop_duplicates()
    removed = original_len - len(df)
    if removed:
        log.append(f"Removed {removed} duplicate rows.")
    else:
        log.append("No duplicate rows found.")

    bad_price = df[TARGET_COL] <= 0
    if bad_price.sum():
        df = df[~bad_price]
        log.append(f"Removed {bad_price.sum()} rows with Price <= 0.")
    else:
        log.append("All Price values are positive — no action needed.")

    bad_area = df["Area"] <= 0
    if bad_area.sum():
        df = df[~bad_area]
        log.append(f"Removed {bad_area.sum()} rows with Area <= 0.")
    else:
        log.append("All Area values are positive — no action needed.")

    for col in ["Bedrooms", "Bathrooms", "Floors"]:
        if col in df.columns:
            outliers = ((df[col] < 1) | (df[col] > 20)).sum()
            if outliers:
                df[col] = df[col].clip(1, 20)
                log.append(f"Clamped {outliers} out-of-range values in '{col}'.")
            else:
                log.append(f"'{col}' values are within expected range — no action needed.")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    log.append("Categorical columns standardised (strip + title-case).")
    log.append(f"Clean dataset: {len(df)} rows x {len(df.columns)} columns.")
    return df.reset_index(drop=True), log


def load_clean_data() -> tuple:
    """Load -> validate -> clean. Returns (clean_df, quality_report, clean_log)."""
    df_raw = load_data()
    report = quality_report(df_raw)
    warnings = validate_schema(df_raw)
    if warnings:
        raise ValueError(f"Schema validation failed: {warnings}")
    df_clean, log = clean_data(df_raw)
    return df_clean, report, log


# ==============================================================================
#
#  BACKEND - analytics.py
#
# ==============================================================================

def kpi_summary(df: pd.DataFrame) -> dict:
    return {
        "total_properties": len(df),
        "avg_price": df["Price"].mean(),
        "median_price": df["Price"].median(),
        "min_price": df["Price"].min(),
        "max_price": df["Price"].max(),
        "avg_area": df["Area"].mean(),
        "median_area": df["Area"].median(),
        "n_locations": df["Location"].nunique(),
        "locations": df["Location"].unique().tolist(),
        "n_conditions": df["Condition"].nunique(),
        "pct_garage": (df["Garage"] == "Yes").mean() * 100,
        "avg_bedrooms": df["Bedrooms"].mean(),
        "avg_bathrooms": df["Bathrooms"].mean(),
        "avg_year_built": df["YearBuilt"].mean(),
        "price_std": df["Price"].std(),
        "price_25": df["Price"].quantile(0.25),
        "price_75": df["Price"].quantile(0.75),
    }


def location_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Location")
        .agg(
            Count=("Price", "count"),
            Avg_Price=("Price", "mean"),
            Median_Price=("Price", "median"),
            Min_Price=("Price", "min"),
            Max_Price=("Price", "max"),
            Avg_Area=("Area", "mean"),
        )
        .round(0)
        .sort_values("Avg_Price", ascending=False)
        .reset_index()
    )


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    order = ["Excellent", "Good", "Fair", "Poor"]
    order = [o for o in order if o in df["Condition"].unique()]
    s = (
        df.groupby("Condition")
        .agg(
            Count=("Price", "count"),
            Avg_Price=("Price", "mean"),
            Median_Price=("Price", "median"),
        )
        .round(0)
        .reindex(order)
        .dropna()
        .reset_index()
    )
    return s


def bedroom_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Bedrooms")
        .agg(Count=("Price", "count"), Avg_Price=("Price", "mean"))
        .round(0)
        .reset_index()
    )


def garage_impact(df: pd.DataFrame) -> dict:
    yes_price = df.loc[df["Garage"] == "Yes", "Price"].mean()
    no_price  = df.loc[df["Garage"] == "No",  "Price"].mean()
    return {
        "garage_yes_avg": yes_price,
        "garage_no_avg": no_price,
        "premium": yes_price - no_price,
        "premium_pct": ((yes_price - no_price) / no_price) * 100,
    }


def price_correlations(df: pd.DataFrame) -> pd.Series:
    num_cols = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
    return df[num_cols].corrwith(df["Price"]).sort_values(ascending=False).round(3)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "Price"]
    return df[num_cols].describe().T.round(2)


def price_segments(df: pd.DataFrame) -> pd.DataFrame:
    q1, q2, q3 = df["Price"].quantile([0.33, 0.66, 1.0])
    bins   = [0, q1, q2, df["Price"].max() + 1]
    labels = ["Budget", "Mid-Range", "Premium"]
    out = df.copy()
    out["Segment"] = pd.cut(out["Price"], bins=bins, labels=labels)
    return out.groupby("Segment", observed=True).agg(
        Count=("Price", "count"),
        Avg_Price=("Price", "mean"),
        Avg_Area=("Area", "mean"),
    ).round(0).reset_index()


def apply_filters(
    df: pd.DataFrame,
    locations=None,
    conditions=None,
    garage=None,
    bedroom_range=None,
    price_range=None,
    area_range=None,
) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    if locations:
        mask &= df["Location"].isin(locations)
    if conditions:
        mask &= df["Condition"].isin(conditions)
    if garage and garage != "All":
        mask &= df["Garage"] == garage
    if bedroom_range:
        mask &= df["Bedrooms"].between(*bedroom_range)
    if price_range:
        mask &= df["Price"].between(*price_range)
    if area_range:
        mask &= df["Area"].between(*area_range)
    return df[mask].reset_index(drop=True)


# ==============================================================================
#
#  BACKEND - eda.py
#
# ==============================================================================

_PRIMARY   = "#4F46E5"
_SECONDARY = "#7C3AED"
_TEAL      = "#0D9488"
_ACCENT    = "#F59E0B"
_BG        = "#F8FAFC"
_GRID      = "#E2E8F0"
_SEQ_PALETTE = px.colors.sequential.Blues
_CAT_PALETTE = [_PRIMARY, _SECONDARY, _TEAL, _ACCENT, "#EC4899", "#10B981"]

_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor=_BG,
    font=dict(family="Segoe UI, sans-serif", size=13),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _apply_layout(fig, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=16, color="#1E293B")),
                      **_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    return fig


def price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="Price", nbins=50,
                       color_discrete_sequence=[_PRIMARY], opacity=0.85, marginal="box")
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return _apply_layout(fig, "House Price Distribution")


def price_vs_area(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="Area", y="Price", color="Location",
                     color_discrete_sequence=_CAT_PALETTE, opacity=0.6,
                     hover_data=["Bedrooms", "Condition"])
    return _apply_layout(fig, "Price vs Area")


def price_by_location(df: pd.DataFrame) -> go.Figure:
    order = (df.groupby("Location")["Price"].median()
               .sort_values(ascending=False).index.tolist())
    fig = px.box(df, x="Location", y="Price", color="Location",
                 color_discrete_sequence=_CAT_PALETTE,
                 category_orders={"Location": order}, notched=True)
    fig.update_traces(boxmean="sd")
    return _apply_layout(fig, "Price Distribution by Location")


def price_by_condition(df: pd.DataFrame) -> go.Figure:
    order = ["Excellent", "Good", "Fair", "Poor"]
    order = [o for o in order if o in df["Condition"].unique()]
    avg = df.groupby("Condition")["Price"].mean().reindex(order).reset_index()
    fig = px.bar(avg, x="Condition", y="Price", color="Condition",
                 color_discrete_sequence=_CAT_PALETTE, text_auto=".2s",
                 category_orders={"Condition": order})
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Average Price by Condition")


def price_by_bedrooms(df: pd.DataFrame) -> go.Figure:
    avg = df.groupby("Bedrooms")["Price"].mean().reset_index()
    fig = px.line(avg, x="Bedrooms", y="Price", markers=True,
                  color_discrete_sequence=[_SECONDARY])
    fig.update_traces(line_width=3, marker_size=8)
    return _apply_layout(fig, "Average Price by Number of Bedrooms")


def price_by_garage(df: pd.DataFrame) -> go.Figure:
    avg = df.groupby("Garage")["Price"].mean().reset_index()
    fig = px.bar(avg, x="Garage", y="Price", color="Garage",
                 color_discrete_sequence=[_TEAL, _PRIMARY], text_auto=".2s")
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Average Price: Garage vs No Garage")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_cols = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "Price"]
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.values.round(2), texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    return _apply_layout(fig, "Correlation Heatmap (Numerical Features)")


def price_by_year(df: pd.DataFrame) -> go.Figure:
    decade = df.copy()
    decade["Decade"] = (decade["YearBuilt"] // 10 * 10).astype(str) + "s"
    avg = decade.groupby("Decade")["Price"].mean().reset_index().sort_values("Decade")
    fig = px.bar(avg, x="Decade", y="Price", color="Price",
                 color_continuous_scale=_SEQ_PALETTE, text_auto=".2s")
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Average Price by Build Decade")


def price_by_floors(df: pd.DataFrame) -> go.Figure:
    avg = df.groupby("Floors")["Price"].mean().reset_index()
    fig = px.bar(avg, x="Floors", y="Price", color="Floors",
                 color_continuous_scale=_SEQ_PALETTE, text_auto=".2s")
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Average Price by Number of Floors")


def location_avg_price(df: pd.DataFrame) -> go.Figure:
    avg = df.groupby("Location")["Price"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(avg, x="Location", y="Price", color="Location",
                 color_discrete_sequence=_CAT_PALETTE, text_auto=".2s")
    fig.update_traces(textposition="outside")
    return _apply_layout(fig, "Average Price by Location")


def price_violin(df: pd.DataFrame) -> go.Figure:
    order = ["Excellent", "Good", "Fair", "Poor"]
    order = [o for o in order if o in df["Condition"].unique()]
    fig = px.violin(df, x="Condition", y="Price", color="Location",
                    color_discrete_sequence=_CAT_PALETTE, box=True,
                    category_orders={"Condition": order})
    return _apply_layout(fig, "Price Distribution by Condition & Location")


def outlier_boxplots(df: pd.DataFrame) -> go.Figure:
    cols = ["Area", "Price", "Bedrooms", "Bathrooms", "Floors"]
    fig = make_subplots(rows=1, cols=len(cols), subplot_titles=cols)
    for i, col in enumerate(cols, start=1):
        fig.add_trace(
            go.Box(y=df[col], name=col,
                   marker_color=_CAT_PALETTE[(i - 1) % len(_CAT_PALETTE)],
                   boxmean="sd"),
            row=1, col=i,
        )
    fig.update_layout(showlegend=False, **_LAYOUT,
                      title=dict(text="Outlier Overview (Box Plots)",
                                 font=dict(size=16, color="#1E293B")))
    return fig


# ==============================================================================
#
#  BACKEND - feature_engineering.py
#
# ==============================================================================

NUMERICAL_FEATURES_EXT = NUMERICAL_FEATURES + ["TotalRooms", "PropertyAge", "AreaPerRoom"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add defensible derived features."""
    df = df.copy()
    df["TotalRooms"]  = df["Bedrooms"] + df["Bathrooms"]
    df["PropertyAge"] = 2025 - df["YearBuilt"]
    df["AreaPerRoom"] = df["Area"] / df["TotalRooms"].replace(0, 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Returns a fitted-ready ColumnTransformer."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES_EXT),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features(df: pd.DataFrame) -> tuple:
    """Apply derived features, return (X, y)."""
    df = add_derived_features(df)
    X = df[NUMERICAL_FEATURES_EXT + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return human-readable feature names after transform."""
    num_names = NUMERICAL_FEATURES_EXT.copy()
    cat_names = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    return num_names + cat_names


# ==============================================================================
#
#  BACKEND - ml_model.py
#
# ==============================================================================

MODEL_PATH   = "models/house_price_model.pkl"
RESULTS_PATH = "models/model_results.pkl"

_COLOURS = {
    "Linear Regression": "#4F46E5",
    "Random Forest":     "#7C3AED",
    "Gradient Boosting": "#0D9488",
}


def _extract_importance(pipeline: Pipeline, feature_names: list, model_name: str) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        importances = importances / (importances.sum() + 1e-9)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    df["Importance"] = df["Importance"].round(4)
    return df


def train_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Train three regression models, evaluate on hold-out test set."""
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=random_state,
                                                    n_jobs=-1, min_samples_leaf=5),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                        max_depth=4, random_state=random_state),
    }

    results = {}
    trained_pipelines = {}

    for name, estimator in candidates.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        estimator),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        results[name] = {
            "MAE":    round(mae,  2),
            "RMSE":   round(rmse, 2),
            "R2":     round(r2,   4),
            "y_test": y_test.values,
            "y_pred": y_pred,
        }
        trained_pipelines[name] = pipeline

    best_name     = max(results, key=lambda k: results[k]["R2"])
    best_pipeline = trained_pipelines[best_name]
    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    importance_df = _extract_importance(best_pipeline, feature_names, best_name)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    full_results = {
        "metrics":            results,
        "best_model":         best_name,
        "feature_importance": importance_df,
        "X_test":             X_test,
        "y_test":             y_test.values,
        "y_pred_best":        results[best_name]["y_pred"],
        "feature_names":      feature_names,
    }
    joblib.dump(full_results, RESULTS_PATH)
    return full_results


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def load_results():
    if os.path.exists(RESULTS_PATH):
        return joblib.load(RESULTS_PATH)
    return None


def model_comparison_chart(results: dict) -> go.Figure:
    names     = list(results.keys())
    r2_vals   = [results[n]["R2"]   for n in names]
    rmse_vals = [results[n]["RMSE"] for n in names]
    mae_vals  = [results[n]["MAE"]  for n in names]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["R² (higher is better)",
                                        "RMSE (lower is better)",
                                        "MAE (lower is better)"])
    colours = [_COLOURS.get(n, "#64748B") for n in names]

    fig.add_trace(go.Bar(x=names, y=r2_vals, marker_color=colours,
                         text=[f"{v:.3f}" for v in r2_vals],
                         textposition="outside", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=rmse_vals, marker_color=colours,
                         text=[f"{v:,.0f}" for v in rmse_vals],
                         textposition="outside", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=mae_vals, marker_color=colours,
                         text=[f"{v:,.0f}" for v in mae_vals],
                         textposition="outside", showlegend=False), row=1, col=3)

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font=dict(family="Segoe UI, sans-serif", size=13),
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text="Model Comparison", font=dict(size=16, color="#1E293B")),
    )
    fig.update_xaxes(showgrid=False)
    return fig


def actual_vs_predicted_chart(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    perfect = np.linspace(y_test.min(), y_test.max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                             marker=dict(color="#4F46E5", opacity=0.5, size=5),
                             name="Predictions"))
    fig.add_trace(go.Scatter(x=perfect, y=perfect, mode="lines",
                             line=dict(color="#EF4444", dash="dash", width=2),
                             name="Perfect Prediction"))
    fig.update_layout(
        title=dict(text=f"Actual vs Predicted — {model_name}",
                   font=dict(size=16, color="#1E293B")),
        xaxis_title="Actual Price", yaxis_title="Predicted Price",
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font=dict(family="Segoe UI, sans-serif", size=13),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def residual_chart(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    residuals = y_test - y_pred
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuals vs Predicted", "Residual Distribution"])
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                             marker=dict(color="#7C3AED", opacity=0.5, size=4),
                             name="Residual"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, nbinsx=40,
                               marker_color="#0D9488", opacity=0.8,
                               name="Distribution"), row=1, col=2)
    fig.update_layout(
        title=dict(text=f"Residual Analysis — {model_name}",
                   font=dict(size=16, color="#1E293B")),
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font=dict(family="Segoe UI, sans-serif", size=13),
        margin=dict(l=40, r=20, t=60, b=40), showlegend=False,
    )
    return fig


def feature_importance_chart(importance_df: pd.DataFrame, model_name: str, top_n: int = 15) -> go.Figure:
    df = importance_df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df["Importance"], y=df["Feature"], orientation="h",
        marker_color="#4F46E5",
        text=[f"{v:.3f}" for v in df["Importance"]], textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"Top Feature Importances — {model_name}",
                   font=dict(size=16, color="#1E293B")),
        xaxis_title="Importance Score",
        paper_bgcolor="white", plot_bgcolor="#F8FAFC",
        font=dict(family="Segoe UI, sans-serif", size=13),
        margin=dict(l=160, r=40, t=50, b=40),
    )
    return fig


# ==============================================================================
#
#  BACKEND - prediction.py
#
# ==============================================================================

def build_input_df(area, bedrooms, bathrooms, floors, year_built,
                   location, condition, garage) -> pd.DataFrame:
    """Create a single-row DataFrame matching the training schema."""
    row = {
        "Area":      area,
        "Bedrooms":  bedrooms,
        "Bathrooms": bathrooms,
        "Floors":    floors,
        "YearBuilt": year_built,
        "Location":  location,
        "Condition": condition,
        "Garage":    garage,
    }
    df = pd.DataFrame([row])
    df = add_derived_features(df)
    return df[NUMERICAL_FEATURES_EXT + CATEGORICAL_FEATURES]


def predict_price(pipeline, input_df: pd.DataFrame) -> float:
    """Return the predicted price for a single-row input DataFrame."""
    prediction = pipeline.predict(input_df)[0]
    return max(prediction, 0.0)


def validate_inputs(area, bedrooms, bathrooms, floors, year_built) -> list:
    """Return a list of validation error messages (empty = OK)."""
    errors = []
    if area <= 0:
        errors.append("Area must be greater than 0.")
    if area > 50_000:
        errors.append("Area seems unrealistically large (> 50,000 sq ft).")
    if not (1 <= bedrooms <= 20):
        errors.append("Bedrooms must be between 1 and 20.")
    if not (1 <= bathrooms <= 20):
        errors.append("Bathrooms must be between 1 and 20.")
    if not (1 <= floors <= 10):
        errors.append("Floors must be between 1 and 10.")
    if not (1800 <= year_built <= 2025):
        errors.append("Year Built must be between 1800 and 2025.")
    return errors


# ==============================================================================
#
#  BACKEND - insights.py
#
# ==============================================================================

def generate_dataset_insights(df: pd.DataFrame) -> list:
    insights = []

    med  = df["Price"].median()
    mean = df["Price"].mean()
    skew = df["Price"].skew()
    if abs(skew) < 0.5:
        insights.append(
            f"House prices are roughly symmetric, with a median of "
            f"${med:,.0f} and a mean of ${mean:,.0f}.")
    elif skew > 0:
        insights.append(
            f"The price distribution is right-skewed (skewness = {skew:.2f}), "
            f"meaning a small number of high-value properties pull the average "
            f"(${mean:,.0f}) above the median (${med:,.0f}).")
    else:
        insights.append(
            f"The price distribution is left-skewed (skewness = {skew:.2f}). "
            f"Mean: ${mean:,.0f} | Median: ${med:,.0f}.")

    corr_area = df["Area"].corr(df["Price"])
    direction = "positive" if corr_area > 0 else "negative"
    insights.append(
        f"Area has a {direction} association with price "
        f"(correlation = {corr_area:.2f}). "
        + ("Larger properties tend to be listed at higher prices." if corr_area > 0.3
           else "The area-price relationship is relatively weak in this dataset."))

    loc_avg = df.groupby("Location")["Price"].mean()
    top_loc = loc_avg.idxmax()
    low_loc = loc_avg.idxmin()
    insights.append(
        f"'{top_loc}' has the highest average price (${loc_avg[top_loc]:,.0f}), "
        f"while '{low_loc}' has the lowest (${loc_avg[low_loc]:,.0f}) in this dataset.")

    order = ["Excellent", "Good", "Fair", "Poor"]
    order = [o for o in order if o in df["Condition"].unique()]
    cond_avg = df.groupby("Condition")["Price"].mean().reindex(order).dropna()
    if len(cond_avg) >= 2:
        best  = cond_avg.idxmax()
        worst = cond_avg.idxmin()
        pct = abs(cond_avg[best] - cond_avg[worst]) / cond_avg[worst] * 100
        insights.append(
            f"Properties rated '{best}' have an average price "
            f"~{pct:.0f}% higher than those rated '{worst}'.")

    if "Yes" in df["Garage"].values and "No" in df["Garage"].values:
        g_yes = df.loc[df["Garage"] == "Yes", "Price"].mean()
        g_no  = df.loc[df["Garage"] == "No",  "Price"].mean()
        diff  = g_yes - g_no
        pct   = abs(diff) / g_no * 100
        direction2 = "higher" if diff > 0 else "lower"
        insights.append(
            f"Properties with a garage have an average price "
            f"~{pct:.1f}% {direction2} than those without a garage.")

    corr_bed = df["Bedrooms"].corr(df["Price"])
    insights.append(
        f"Bedrooms have a correlation of {corr_bed:.2f} with price. "
        + ("More bedrooms are generally associated with higher prices."
           if corr_bed > 0.2 else
           "The number of bedrooms shows a weak association with price in this dataset."))

    corr_yr = df["YearBuilt"].corr(df["Price"])
    if corr_yr > 0.1:
        insights.append(
            f"Newer properties (higher YearBuilt) tend to be associated with "
            f"higher prices (correlation = {corr_yr:.2f}).")
    elif corr_yr < -0.1:
        insights.append(
            f"Older properties (lower YearBuilt) tend to be associated with "
            f"higher prices (correlation = {corr_yr:.2f}).")
    else:
        insights.append(
            f"The year a property was built has little linear association "
            f"with price in this dataset (correlation = {corr_yr:.2f}).")

    return insights


def generate_model_insights(results: dict) -> list:
    insights = []
    best    = results.get("best_model", "Unknown")
    metrics = results.get("metrics", {})

    if best in metrics:
        m = metrics[best]
        insights.append(
            f"The best-performing model is **{best}** with "
            f"R2 = {m['R2']:.3f}, RMSE = ${m['RMSE']:,.0f}, MAE = ${m['MAE']:,.0f}.")
        if m["R2"] >= 0.90:
            insights.append(
                "An R2 of >= 0.90 indicates the model explains the large majority "
                "of variance in house prices on the test set.")
        elif m["R2"] >= 0.75:
            insights.append(
                "An R2 between 0.75 and 0.90 indicates a reasonably strong fit.")
        elif m["R2"] >= 0.30:
            insights.append(
                "The model has moderate predictive power (R2 between 0.30-0.75).")
        else:
            insights.append(
                "The model shows low predictive power (R2 < 0.30). "
                "This is a strong signal that the available features have very weak "
                "relationships with price in this dataset.")
            insights.append(
                "An R2 near 0 or negative means the model performs no better than "
                "simply predicting the mean price for every property. "
                "This is honest — the dashboard does not fabricate accuracy.")

    fi = results.get("feature_importance")
    if fi is not None and len(fi) > 0:
        top3 = fi.head(3)["Feature"].tolist()
        insights.append(
            f"The top features associated with predictions are: "
            + ", ".join(f"**{f}**" for f in top3) + ". "
            "Note: feature importance reflects association with predictions, "
            "not direct causation of house price changes.")

    return insights


def generate_actionable_insights(df: pd.DataFrame, results: dict) -> list:
    insights = []

    loc_avg = df.groupby("Location")["Price"].mean().sort_values(ascending=False)
    if len(loc_avg) >= 2:
        insights.append(
            f"**Location matters most in this dataset.** '{loc_avg.index[0]}' "
            f"averages ${loc_avg.iloc[0]:,.0f} vs '{loc_avg.index[-1]}' at "
            f"${loc_avg.iloc[-1]:,.0f}.")

    med_area  = df["Area"].median()
    med_price = df["Price"].median()
    insights.append(
        f"The median property is {med_area:,.0f} sq ft at ${med_price:,.0f}. "
        "Consider area-to-price ratio when comparing properties.")

    order = ["Excellent", "Good", "Fair", "Poor"]
    order = [o for o in order if o in df["Condition"].unique()]
    cond_avg = df.groupby("Condition")["Price"].mean().reindex(order).dropna()
    if len(cond_avg) >= 2:
        top = cond_avg.idxmax()
        bot = cond_avg.idxmin()
        pct = (cond_avg[top] - cond_avg[bot]) / cond_avg[bot] * 100
        insights.append(
            f"Property condition is associated with a price difference of up to "
            f"~{pct:.0f}% between '{top}' and '{bot}' rated properties.")

    best = results.get("best_model", "")
    if best:
        m    = results["metrics"].get(best, {})
        r2   = m.get("R2", 0)
        rmse = m.get("RMSE", 0)
        insights.append(
            f"The ML model ({best}, R2 = {r2:.3f}) estimates prices with an "
            f"average error of about ${rmse:,.0f}. "
            "Use predictions as one input among many — not as a final price.")

    insights.append(
        "**Remember:** These insights are based on historical data patterns. "
        "They do not guarantee future prices and should not be the sole basis "
        "for financial or property decisions.")
    return insights


# ==============================================================================
#
#  FRONTEND - app.py  (Streamlit Dashboard)
#
# ==============================================================================

st.set_page_config(
    page_title="HOUSEPRICEAI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background: #F0F4F8; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px; border-radius: 8px; cursor: pointer;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.15); }
.card {
    background: white; border-radius: 16px; padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.card-sm {
    background: white; border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07); margin-bottom: 12px;
}
.kpi-card {
    background: white; border-radius: 16px; padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.04);
    border-left: 5px solid #4F46E5; text-align: center;
}
.kpi-card.teal   { border-left-color: #0D9488; }
.kpi-card.purple { border-left-color: #7C3AED; }
.kpi-card.amber  { border-left-color: #D97706; }
.kpi-card.green  { border-left-color: #059669; }
.kpi-card.rose   { border-left-color: #E11D48; }
.kpi-label { font-size: 12px; font-weight: 600; color: #64748B;
             text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #1E293B; line-height: 1.1; }
.kpi-sub   { font-size: 12px; color: #94A3B8; margin-top: 4px; }
.hero {
    background: linear-gradient(135deg, #1E1B4B 0%, #4338CA 60%, #0D9488 100%);
    border-radius: 20px; padding: 48px 40px; color: white;
    margin-bottom: 24px; text-align: center;
}
.hero h1 { font-size: 48px; font-weight: 800; letter-spacing: -1px; margin: 0 0 8px; }
.hero .subtitle { font-size: 20px; opacity: 0.9; margin: 0 0 6px; font-weight: 300; }
.hero .tagline  { font-size: 14px; opacity: 0.7; }
.section-title {
    font-size: 20px; font-weight: 700; color: #1E293B;
    border-left: 4px solid #4F46E5; padding-left: 12px;
    margin: 24px 0 16px;
}
.pred-card {
    background: linear-gradient(135deg, #1E1B4B 0%, #4338CA 100%);
    border-radius: 20px; padding: 40px 32px; color: white;
    text-align: center; box-shadow: 0 8px 32px rgba(67,56,202,0.35);
}
.pred-label { font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase;
              opacity: 0.75; margin-bottom: 12px; }
.pred-price { font-size: 52px; font-weight: 800; letter-spacing: -2px;
              line-height: 1; margin-bottom: 20px; }
.pred-meta  { font-size: 14px; opacity: 0.85; line-height: 1.8; }
.pred-badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    border-radius: 20px; padding: 4px 14px; font-size: 12px;
    margin-top: 12px; font-weight: 600;
}
.badge { display: inline-block; border-radius: 20px; padding: 3px 12px;
         font-size: 12px; font-weight: 600; }
.badge-blue  { background: #EEF2FF; color: #4F46E5; }
.badge-green { background: #D1FAE5; color: #059669; }
.badge-amber { background: #FEF3C7; color: #D97706; }
.badge-red   { background: #FEE2E2; color: #DC2626; }
.insight-box {
    background: #EEF2FF; border-left: 4px solid #4F46E5;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
    font-size: 14px; color: #1E293B; line-height: 1.6;
}
.insight-box.green { background: #D1FAE5; border-left-color: #059669; }
.insight-box.amber { background: #FEF3C7; border-left-color: #D97706; }
.insight-box.red   { background: #FEE2E2; border-left-color: #DC2626; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
[data-testid="stMetricValue"] { font-size: 22px !important; }
.divider { border: none; border-top: 1px solid #E2E8F0; margin: 20px 0; }
.rai-box {
    background: #FFF7ED; border: 1px solid #FED7AA;
    border-radius: 12px; padding: 20px 24px;
    font-size: 13px; color: #92400E; line-height: 1.7;
}
.tech-pill {
    display: inline-block; background: #EEF2FF; color: #4338CA;
    border-radius: 20px; padding: 4px 14px; font-size: 13px;
    font-weight: 500; margin: 3px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data():
    return load_clean_data()


@st.cache_resource(show_spinner=False)
def get_models(df):
    existing = load_results()
    if existing is not None:
        pipeline = load_model()
        return pipeline, existing
    results  = train_models(df)
    pipeline = load_model()
    return pipeline, results


try:
    df, quality_rpt, clean_log = get_data()
except FileNotFoundError as e:
    st.error(f"Dataset not found: {e}")
    st.stop()
except ValueError as e:
    st.error(f"Schema error: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading data: {e}")
    st.stop()

with st.spinner("Training models (first run only)..."):
    try:
        pipeline, model_results = get_models(df)
    except Exception as e:
        st.error(f"Model training error: {e}")
        st.stop()

kpi = kpi_summary(df)

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:40px;'>🏠</div>
        <div style='font-size:20px; font-weight:800; letter-spacing:-0.5px;'>HOUSEPRICEAI</div>
        <div style='font-size:11px; opacity:0.7; margin-top:4px;'>Powered by Machine Learning</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.2); margin:12px 0;'>
    """, unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        ["🏠  Dashboard", "📊  Analytics", "🔍  Data Explorer",
         "🤖  Prediction", "📈  Model Performance", "💡  Insights", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.2); margin:16px 0;'>",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:12px; opacity:0.65; line-height:1.8;'>
        🗂️ {kpi['total_properties']:,} properties<br>
        📍 {kpi['n_locations']} locations<br>
        🤖 Best: {model_results['best_model']}<br>
        📊 R² = {model_results['metrics'][model_results['best_model']]['R2']:.3f}
    </div>
    """, unsafe_allow_html=True)


def kpi_row(kpi: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Total Properties</div>
            <div class='kpi-value'>{kpi['total_properties']:,}</div>
            <div class='kpi-sub'>in dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card teal'>
            <div class='kpi-label'>Average Price</div>
            <div class='kpi-value'>${kpi['avg_price']:,.0f}</div>
            <div class='kpi-sub'>mean listing</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card purple'>
            <div class='kpi-label'>Median Price</div>
            <div class='kpi-value'>${kpi['median_price']:,.0f}</div>
            <div class='kpi-sub'>50th percentile</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-card amber'>
            <div class='kpi-label'>Avg Area</div>
            <div class='kpi-value'>{kpi['avg_area']:,.0f}</div>
            <div class='kpi-sub'>sq ft</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class='kpi-card green'>
            <div class='kpi-label'>Locations</div>
            <div class='kpi-value'>{kpi['n_locations']}</div>
            <div class='kpi-sub'>{", ".join(kpi['locations'])}</div>
        </div>""", unsafe_allow_html=True)


# ── PAGE: DASHBOARD ───────────────────────────────────────────────────────────
if nav == "🏠  Dashboard":
    st.markdown("""
    <div class='hero'>
        <h1>🏠 HOUSEPRICEAI</h1>
        <div class='subtitle'>Intelligent House Price Prediction &amp; Analytics</div>
        <div class='tagline'>Data-driven insights for understanding and estimating residential property prices</div>
    </div>
    """, unsafe_allow_html=True)

    kpi_row(kpi)

    st.markdown("<div class='section-title'>📊 Price Overview</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(price_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(location_avg_price(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(price_by_condition(df), use_container_width=True)
    with col4:
        st.plotly_chart(price_vs_area(df), use_container_width=True)

    st.markdown("<div class='section-title'>🏆 Location Summary</div>", unsafe_allow_html=True)
    loc_df = location_summary(df)
    loc_df["Avg_Price"]    = loc_df["Avg_Price"].map("${:,.0f}".format)
    loc_df["Median_Price"] = loc_df["Median_Price"].map("${:,.0f}".format)
    loc_df["Min_Price"]    = loc_df["Min_Price"].map("${:,.0f}".format)
    loc_df["Max_Price"]    = loc_df["Max_Price"].map("${:,.0f}".format)
    loc_df["Avg_Area"]     = loc_df["Avg_Area"].map("{:,.0f} sqft".format)
    st.dataframe(loc_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>💡 Quick Insights</div>", unsafe_allow_html=True)
    for ins in generate_dataset_insights(df)[:4]:
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)


# ── PAGE: ANALYTICS ───────────────────────────────────────────────────────────
elif nav == "📊  Analytics":
    st.markdown("<div class='hero' style='padding:32px 40px;'>"
                "<h1 style='font-size:32px;'>📊 Analytics</h1>"
                "<div class='tagline'>Interactive exploration of house price patterns</div>"
                "</div>", unsafe_allow_html=True)

    with st.expander("🔧 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_locations = st.multiselect(
                "Location", df["Location"].unique().tolist(),
                default=df["Location"].unique().tolist(), key="an_loc")
        with fc2:
            sel_conditions = st.multiselect(
                "Condition", df["Condition"].unique().tolist(),
                default=df["Condition"].unique().tolist(), key="an_cond")
        with fc3:
            sel_garage = st.selectbox("Garage", ["All", "Yes", "No"], key="an_gar")

        fc4, fc5 = st.columns(2)
        with fc4:
            price_min, price_max = int(df["Price"].min()), int(df["Price"].max())
            sel_price = st.slider("Price Range ($)", price_min, price_max,
                                  (price_min, price_max), step=5000, key="an_price")
        with fc5:
            bed_min, bed_max = int(df["Bedrooms"].min()), int(df["Bedrooms"].max())
            sel_bed = st.slider("Bedrooms", bed_min, bed_max,
                                (bed_min, bed_max), key="an_bed")

    fdf = apply_filters(df, sel_locations, sel_conditions,
                        sel_garage if sel_garage != "All" else None,
                        sel_bed, sel_price)

    if len(fdf) == 0:
        st.warning("No properties match the current filters. Please adjust the selection.")
        st.stop()

    st.markdown(f"<div class='card-sm'>Showing <b>{len(fdf):,}</b> of "
                f"<b>{len(df):,}</b> properties</div>", unsafe_allow_html=True)

    kpi_row(kpi_summary(fdf))

    st.markdown("<div class='section-title'>Price Patterns</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(price_by_location(fdf), use_container_width=True)
    with col2:
        st.plotly_chart(price_by_condition(fdf), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(price_vs_area(fdf), use_container_width=True)
    with col4:
        st.plotly_chart(price_by_bedrooms(fdf), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(price_by_garage(fdf), use_container_width=True)
    with col6:
        st.plotly_chart(price_by_floors(fdf), use_container_width=True)

    st.markdown("<div class='section-title'>Build Year & Correlations</div>",
                unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(price_by_year(fdf), use_container_width=True)
    with col8:
        st.plotly_chart(correlation_heatmap(fdf), use_container_width=True)

    st.markdown("<div class='section-title'>Distributions</div>", unsafe_allow_html=True)
    st.plotly_chart(price_violin(fdf), use_container_width=True)
    st.plotly_chart(outlier_boxplots(fdf), use_container_width=True)


# ── PAGE: DATA EXPLORER ───────────────────────────────────────────────────────
elif nav == "🔍  Data Explorer":
    st.markdown("<div class='hero' style='padding:32px 40px;'>"
                "<h1 style='font-size:32px;'>🔍 Data Explorer</h1>"
                "<div class='tagline'>Understand the raw dataset structure and quality</div>"
                "</div>", unsafe_allow_html=True)

    oc1, oc2, oc3, oc4 = st.columns(4)
    with oc1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Rows</div>"
                    f"<div class='kpi-value'>{quality_rpt['n_rows']:,}</div></div>",
                    unsafe_allow_html=True)
    with oc2:
        st.markdown(f"<div class='kpi-card teal'><div class='kpi-label'>Columns</div>"
                    f"<div class='kpi-value'>{quality_rpt['n_cols']}</div></div>",
                    unsafe_allow_html=True)
    with oc3:
        total_miss = sum(quality_rpt['missing'].values())
        st.markdown(f"<div class='kpi-card purple'><div class='kpi-label'>Missing Values</div>"
                    f"<div class='kpi-value'>{total_miss}</div></div>",
                    unsafe_allow_html=True)
    with oc4:
        st.markdown(f"<div class='kpi-card amber'><div class='kpi-label'>Duplicates</div>"
                    f"<div class='kpi-value'>{quality_rpt['duplicates']}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Column Quality Report</div>",
                unsafe_allow_html=True)
    miss_df = pd.DataFrame({
        "Column":    list(quality_rpt["dtypes"].keys()),
        "Type":      list(quality_rpt["dtypes"].values()),
        "Missing":   [quality_rpt["missing"][c] for c in quality_rpt["dtypes"]],
        "Missing %": [quality_rpt["missing_pct"][c] for c in quality_rpt["dtypes"]],
        "Unique":    [quality_rpt["unique_values"][c] for c in quality_rpt["dtypes"]],
    })
    st.dataframe(miss_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Data Cleaning Log</div>",
                unsafe_allow_html=True)
    for step in clean_log:
        icon = "✅" if "no action" in step.lower() or "no duplicate" in step.lower() else "🔧"
        st.markdown(f"<div class='insight-box green'>{icon} {step}</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Descriptive Statistics</div>",
                unsafe_allow_html=True)
    st.dataframe(descriptive_stats(df), use_container_width=True)

    st.markdown("<div class='section-title'>Categorical Column Values</div>",
                unsafe_allow_html=True)
    for col, vals in quality_rpt["categorical_uniques"].items():
        badges = " ".join([f"<span class='badge badge-blue'>{v}</span>" for v in vals])
        st.markdown(f"<div class='card-sm'><b>{col}</b>: {badges}</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Data Preview</div>", unsafe_allow_html=True)
    n_rows = st.slider("Rows to display", 10, 200, 50)
    st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>Price Segments</div>", unsafe_allow_html=True)
    seg_df = price_segments(df)
    seg_df["Avg_Price"] = seg_df["Avg_Price"].map("${:,.0f}".format)
    seg_df["Avg_Area"]  = seg_df["Avg_Area"].map("{:,.0f} sqft".format)
    st.dataframe(seg_df, use_container_width=True, hide_index=True)


# ── PAGE: PREDICTION ──────────────────────────────────────────────────────────
elif nav == "🤖  Prediction":
    st.markdown("<div class='hero' style='padding:32px 40px;'>"
                "<h1 style='font-size:32px;'>🤖 Price Prediction</h1>"
                "<div class='tagline'>Estimate a property price using our trained ML model</div>"
                "</div>", unsafe_allow_html=True)

    _pred_r2 = model_results["metrics"][model_results["best_model"]]["R2"]
    if _pred_r2 < 0.30:
        st.info(
            f"ℹ️ **Model note:** The best model (R² = {_pred_r2:.3f}) has low predictive power "
            "because the prices in this dataset are near-randomly distributed relative to all "
            "features. The prediction shown below is the model's output — it will be close to "
            "the dataset mean (~$537K) for most inputs. See **Model Performance** for full details."
        )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("<div class='section-title'>Property Details</div>",
                    unsafe_allow_html=True)
        with st.container():
            area = st.number_input("📐 Area (sq ft)", min_value=100, max_value=50000,
                                   value=2500, step=50)
            c1, c2 = st.columns(2)
            with c1:
                bedrooms   = st.number_input("🛏️ Bedrooms",  1, 20, 3)
                floors     = st.number_input("🏢 Floors",    1, 10, 2)
                location   = st.selectbox("📍 Location",
                                          sorted(df["Location"].unique().tolist()))
            with c2:
                bathrooms  = st.number_input("🚿 Bathrooms", 1, 20, 2)
                year_built = st.number_input("🗓️ Year Built", 1800, 2025, 2000)
                condition  = st.selectbox("🏠 Condition",
                                          ["Excellent", "Good", "Fair", "Poor"])
            garage = st.radio("🚗 Garage", ["Yes", "No"], horizontal=True)
            predict_btn = st.button("🔮 Predict Price", type="primary",
                                    use_container_width=True)

    with right:
        st.markdown("<div class='section-title'>Prediction Result</div>",
                    unsafe_allow_html=True)
        if predict_btn:
            errors = validate_inputs(area, bedrooms, bathrooms, floors, year_built)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                try:
                    input_df = build_input_df(
                        area=float(area), bedrooms=int(bedrooms),
                        bathrooms=int(bathrooms), floors=int(floors),
                        year_built=int(year_built),
                        location=location, condition=condition, garage=garage,
                    )
                    price = predict_price(pipeline, input_df)
                    best  = model_results["best_model"]
                    bm    = model_results["metrics"][best]

                    st.markdown(f"""
                    <div class='pred-card'>
                        <div class='pred-label'>Estimated House Price</div>
                        <div class='pred-price'>${price:,.0f}</div>
                        <div class='pred-meta'>
                            Model: <b>{best}</b><br>
                            R² = {bm['R2']:.3f} &nbsp;|&nbsp; RMSE = ${bm['RMSE']:,.0f}<br>
                            MAE = ${bm['MAE']:,.0f}
                        </div>
                        <div class='pred-badge'>📊 Based on {len(df):,} training properties</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='card-sm'><b>Input Summary</b><br><br>"
                                + f"📐 Area: {area:,} sq ft &nbsp;|&nbsp; "
                                + f"🛏️ {bedrooms} beds &nbsp;|&nbsp; 🚿 {bathrooms} baths<br>"
                                + f"🏢 {floors} floor(s) &nbsp;|&nbsp; "
                                + f"🗓️ Built {year_built}<br>"
                                + f"📍 {location} &nbsp;|&nbsp; 🏠 {condition} &nbsp;|&nbsp; 🚗 Garage: {garage}"
                                + "</div>", unsafe_allow_html=True)

                    st.markdown("""
                    <div class='rai-box' style='margin-top:12px;'>
                        ⚠️ <b>Disclaimer:</b> This is an ML estimate based on historical data.
                        It is not a formal appraisal. Do not use it as the sole basis for
                        financial or property decisions. Actual prices may differ.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:60px 24px; color:#94A3B8;'>
                <div style='font-size:48px;'>🏠</div>
                <div style='font-size:16px; margin-top:12px;'>
                    Fill in the property details on the left<br>and click <b>Predict Price</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='card-sm' style='margin-top:8px; font-size:13px; color:#64748B;'>
        🤖 <b>Model in use:</b> {model_results['best_model']} &nbsp;|&nbsp;
        Selected automatically based on highest R² on hold-out test set.
    </div>
    """, unsafe_allow_html=True)


# ── PAGE: MODEL PERFORMANCE ───────────────────────────────────────────────────
elif nav == "📈  Model Performance":
    st.markdown("<div class='hero' style='padding:32px 40px;'>"
                "<h1 style='font-size:32px;'>📈 Model Performance</h1>"
                "<div class='tagline'>Transparent evaluation of all trained models</div>"
                "</div>", unsafe_allow_html=True)

    metrics = model_results["metrics"]
    best    = model_results["best_model"]
    best_r2 = metrics[best]["R2"]

    if best_r2 < 0.30:
        st.markdown("""
        <div class='rai-box' style='margin-bottom:20px;'>
            <b>📊 Dataset Signal Analysis</b><br><br>
            The trained models show low R² values on this dataset. This is <b>not a bug</b>
            — it reflects a genuine property of the data: the prices in this dataset have
            near-zero correlation with all available features.<br><br>
            <b>What this means:</b> The prices appear to have been assigned independently
            of the features — a pattern characteristic of synthetically generated benchmark
            datasets. The dashboard reports honest metrics rather than fabricating accuracy.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model Comparison</div>",
                unsafe_allow_html=True)
    rows = []
    for name, m in metrics.items():
        badge = "⭐ Selected" if name == best else ""
        rows.append({"Model": name, "MAE ($)": f"${m['MAE']:,.0f}",
                     "RMSE ($)": f"${m['RMSE']:,.0f}", "R²": m["R2"], "": badge})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.markdown("""<div class='card-sm'>
            <b>MAE – Mean Absolute Error</b><br>
            <span style='color:#64748B;font-size:13px;'>
            Average absolute difference between predicted and actual prices.
            Lower is better. Same units as price.
            </span></div>""", unsafe_allow_html=True)
    with ec2:
        st.markdown("""<div class='card-sm'>
            <b>RMSE – Root Mean Squared Error</b><br>
            <span style='color:#64748B;font-size:13px;'>
            Penalises large errors more heavily than MAE.
            Lower is better. Same units as price.
            </span></div>""", unsafe_allow_html=True)
    with ec3:
        st.markdown("""<div class='card-sm'>
            <b>R² – Coefficient of Determination</b><br>
            <span style='color:#64748B;font-size:13px;'>
            Proportion of price variance explained by the model.
            Range: 0–1. Higher is better.
            </span></div>""", unsafe_allow_html=True)

    st.plotly_chart(model_comparison_chart(metrics), use_container_width=True)

    st.markdown(f"<div class='section-title'>Best Model Deep Dive: {best}</div>",
                unsafe_allow_html=True)
    y_test = model_results["y_test"]
    y_pred = model_results["y_pred_best"]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(actual_vs_predicted_chart(y_test, y_pred, best),
                        use_container_width=True)
    with c2:
        st.plotly_chart(residual_chart(y_test, y_pred, best),
                        use_container_width=True)

    fi_df = model_results.get("feature_importance")
    if fi_df is not None and len(fi_df) > 0:
        st.markdown("<div class='section-title'>Feature Importance</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(feature_importance_chart(fi_df, best), use_container_width=True)
        st.markdown("""
        <div class='insight-box amber'>
            ⚠️ <b>Important:</b> Feature importance reflects each feature's
            contribution to the model's predictions — not a causal relationship.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Top Features Table</div>",
                    unsafe_allow_html=True)
        st.dataframe(fi_df.head(15), use_container_width=True, hide_index=True)


# ── PAGE: INSIGHTS ────────────────────────────────────────────────────────────
elif nav == "💡  Insights":
    st.markdown("<div class='hero' style='padding:32px 40px;'>"
                "<h1 style='font-size:32px;'>💡 Insights</h1>"
                "<div class='tagline'>Data-driven observations from the dataset and ML model</div>"
                "</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Dataset Insights</div>",
                unsafe_allow_html=True)
    for ins in generate_dataset_insights(df):
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model Insights</div>",
                unsafe_allow_html=True)
    for ins in generate_model_insights(model_results):
        st.markdown(f"<div class='insight-box green'>{ins}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Actionable Insights</div>",
                unsafe_allow_html=True)
    for ins in generate_actionable_insights(df, model_results):
        cls = "red" if "Remember" in ins else "amber"
        st.markdown(f"<div class='insight-box {cls}'>{ins}</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Numerical Feature Correlations with Price</div>",
                unsafe_allow_html=True)
    corr_s  = price_correlations(df)
    corr_df = corr_s.reset_index()
    corr_df.columns = ["Feature", "Pearson r with Price"]
    st.dataframe(corr_df, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class='insight-box'>
        📌 Pearson r ranges from -1 (perfect negative) to +1 (perfect positive).
        Values near 0 indicate little linear relationship.
        <b>Correlation does not imply causation.</b>
    </div>
    """, unsafe_allow_html=True)

    gi = garage_impact(df)
    st.markdown("<div class='section-title'>Garage Price Impact</div>",
                unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Price — Garage</div>"
                    f"<div class='kpi-value'>${gi['garage_yes_avg']:,.0f}</div></div>",
                    unsafe_allow_html=True)
    with g2:
        st.markdown(f"<div class='kpi-card teal'><div class='kpi-label'>Avg Price — No Garage</div>"
                    f"<div class='kpi-value'>${gi['garage_no_avg']:,.0f}</div></div>",
                    unsafe_allow_html=True)
    with g3:
        direction = "premium" if gi["premium"] > 0 else "discount"
        st.markdown(f"<div class='kpi-card purple'><div class='kpi-label'>Garage {direction.title()}</div>"
                    f"<div class='kpi-value'>${abs(gi['premium']):,.0f}</div>"
                    f"<div class='kpi-sub'>({gi['premium_pct']:+.1f}%)</div></div>",
                    unsafe_allow_html=True)


# ── PAGE: ABOUT ───────────────────────────────────────────────────────────────
elif nav == "ℹ️  About":
    st.markdown("""
    <div class='hero'>
        <h1>🏠 HOUSEPRICEAI</h1>
        <div class='subtitle'>Intelligent House Price Prediction &amp; Analytics Dashboard</div>
        <div class='tagline'>Data Analytics + AI/ML Portfolio Project</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card'>
            <div class='section-title' style='margin-top:0;'>📋 Problem Statement</div>
            <p style='color:#475569;font-size:14px;line-height:1.7;'>
            House prices vary depending on property characteristics such as size, location,
            condition, and other attributes. Raw house-price data alone does not provide
            clear decision-making information. This project transforms raw property data
            into actionable insights and ML-powered price estimates.
            </p>
            <div class='section-title'>🎯 Objectives</div>
            <ul style='color:#475569;font-size:14px;line-height:2;'>
                <li>Understand house price distributions and patterns</li>
                <li>Identify features associated with price variation</li>
                <li>Compare prices across locations and conditions</li>
                <li>Build and evaluate ML regression models</li>
                <li>Provide interactive price estimation</li>
                <li>Generate dynamic, data-driven insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='card'>
            <div class='section-title' style='margin-top:0;'>🗂️ Dataset Overview</div>
            <ul style='color:#475569;font-size:14px;line-height:2;'>
                <li><b>File:</b> House Price Prediction Dataset.csv</li>
                <li><b>Rows:</b> {len(df):,}</li>
                <li><b>Columns:</b> {len(df.columns)}</li>
                <li><b>Features:</b> Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage</li>
                <li><b>Target:</b> Price</li>
                <li><b>Missing values:</b> 0</li>
                <li><b>Duplicates:</b> 0</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <div class='section-title' style='margin-top:0;'>🛠️ Technology Stack</div>
            <div style='margin-top:8px;'>
                <span class='tech-pill'>Python 3.11+</span>
                <span class='tech-pill'>Streamlit</span>
                <span class='tech-pill'>Pandas</span>
                <span class='tech-pill'>NumPy</span>
                <span class='tech-pill'>Scikit-learn</span>
                <span class='tech-pill'>Plotly</span>
                <span class='tech-pill'>Joblib</span>
            </div>
            <div class='section-title'>🤖 ML Models Evaluated</div>
            <ul style='color:#475569;font-size:14px;line-height:2;'>
                <li>Linear Regression</li>
                <li>Random Forest Regressor</li>
                <li>Gradient Boosting Regressor</li>
            </ul>
            <div class='section-title'>🔄 Analytics Workflow</div>
            <p style='color:#475569;font-size:13px;line-height:1.9;'>
                Data Loading → Validation → Cleaning → EDA →
                Feature Engineering → Model Training → Evaluation →
                Persistence → Prediction → Insights
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <div class='section-title' style='margin-top:0;'>📊 Dashboard Sections</div>
            <ul style='color:#475569;font-size:14px;line-height:2;'>
                <li>🏠 <b>Dashboard</b> – KPIs, overview charts, quick insights</li>
                <li>📊 <b>Analytics</b> – Filtered interactive exploration</li>
                <li>🔍 <b>Data Explorer</b> – Schema, quality, statistics</li>
                <li>🤖 <b>Prediction</b> – ML price estimation form</li>
                <li>📈 <b>Model Performance</b> – Metrics, charts, feature importance</li>
                <li>💡 <b>Insights</b> – Dynamic data-driven observations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🛡️ Responsible AI</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='rai-box'>
        <b>🔶 Responsible AI Notice</b><br><br>
        <ul style='margin:0; padding-left:20px; line-height:2;'>
            <li>Predictions are <b>statistical estimates</b>, not formal appraisals.</li>
            <li>Results depend on the quality and representativeness of historical training data.</li>
            <li>Historical patterns do not guarantee future property prices.</li>
            <li>Predictions should <b>not be the sole basis</b> for financial or property decisions.</li>
            <li>Feature importance reflects association with model predictions — it does <b>not imply causation</b>.</li>
            <li>Risk of bias exists if training data is incomplete or unrepresentative.</li>
            <li>No sensitive personal attributes are used in this model.</li>
            <li>Always consult qualified real estate professionals for major property decisions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>⚠️ Limitations</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box amber'>
        <ul style='margin:0; padding-left:20px; line-height:2;'>
            <li>Model accuracy is bounded by the features available in this dataset.</li>
            <li>Factors such as neighbourhood amenities, school ratings, or proximity to
                transport are not included and may significantly affect real-world prices.</li>
            <li>The model is trained on a single static dataset; prices change over time.</li>
            <li>Predictions outside the training data distribution may be less reliable.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr style='border:none;border-top:1px solid #E2E8F0; margin:32px 0 16px;'>
    <div style='text-align:center; font-size:13px; color:#94A3B8;'>
        HOUSEPRICEAI &nbsp;·&nbsp; Data Analytics + AI/ML Portfolio Project
        &nbsp;·&nbsp; Built with Python, Streamlit &amp; Scikit-learn
    </div>
    """, unsafe_allow_html=True)
