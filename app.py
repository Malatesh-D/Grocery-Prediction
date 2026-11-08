"""GrocerySpend AI — Household Grocery Expenditure Intelligence Dashboard.

An end-to-end Machine Learning web application to predict monthly grocery
expenditure in Indian Rupees (INR) with data exploration, multi-model
benchmarking, interactive diagnostics, and spending pattern insights.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_csv
from src.predict import ModelArtifactError, load_artifact, predict_expenditure, validate_inputs
from src.train_model import train_frame
from src.utils import BOUNDS, DATA_PATH, FEATURES, METADATA_PATH, MODEL_PATH, OUTPUT_DIR, TARGET, inr
from src.validate_data import FEATURE_LABELS, map_columns, validate_training_data

# ---------------------------------------------------------------------------
# Page Configuration & Modern Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GrocerySpend AI — Household Expenditure Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --navy: #0f172a;
    --slate: #475569;
    --light-bg: #f8fafc;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --success: #16a34a;
    --warning: #d97706;
}

/* Base typography and background */
.stApp {
    background-color: var(--light-bg);
    color: var(--navy);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Custom Header banner */
.hero-container {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #38bdf8 100%);
    color: white;
    padding: 1.75rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.2);
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    color: #ffffff !important;
}
.hero-subtitle {
    font-size: 1rem;
    color: #e0f2fe !important;
    margin-top: 0.35rem;
    max-width: 850px;
}
.hero-badges {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.85rem;
    flex-wrap: wrap;
}
.badge-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-synthetic {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}
.badge-real {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}
.badge-model {
    background: rgba(255, 255, 255, 0.2);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Modern Card Layout */
.custom-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 1.25rem;
}
.custom-card h3 {
    margin-top: 0;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--navy);
}

/* =========================================================================
   STREAMLIT METRICS (KPI Cards) - CRISP HIGH CONTRAST
   ========================================================================= */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}
div[data-testid="stMetric"] * {
    color: #0f172a !important;
}
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricLabel"] span,
div[data-testid="stMetricLabel"] label {
    color: #475569 !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    opacity: 1 !important;
}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] span {
    color: #0f172a !important;
    font-size: 1.75rem !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] * {
    font-weight: 700 !important;
}

/* =========================================================================
   STREAMLIT ALERTS (Warning, Info, Success, Error) - CRISP HIGH CONTRAST
   ========================================================================= */
div[data-testid="stAlert"],
div.stAlert,
div[data-baseweb="notification"] {
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
div[data-testid="stAlert"] *,
div.stAlert *,
div[data-testid="stAlert"] p,
div.stAlert p,
div[data-testid="stAlert"] span,
div.stAlert span,
div[data-testid="stAlert"] div,
div.stAlert div,
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    line-height: 1.5 !important;
    opacity: 1 !important;
}

/* Warning alert container & text */
div[data-testid="stAlert"]:has(svg),
div.stAlert {
    background-color: #fefce8 !important;
    border: 2px solid #eab308 !important;
}
div[data-testid="stAlert"]:has(svg) *,
div.stAlert * {
    color: #854d0e !important;
    font-weight: 700 !important;
}

/* =========================================================================
   UNIVERSAL TEXT VISIBILITY
   ========================================================================= */
.stMarkdown,
.stMarkdown p,
.stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #1e293b !important;
    opacity: 1 !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-weight: 800 !important;
}

/* Prediction highlight banner */
.prediction-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #86efac;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.prediction-label {
    font-size: 0.9rem;
    font-weight: 700;
    color: #166534;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.prediction-value {
    font-size: 2.75rem;
    font-weight: 900;
    color: #14532d;
    margin: 0.25rem 0;
}
.prediction-subtext {
    font-size: 0.88rem;
    color: #15803d;
}

/* Sidebar clean styling with guaranteed high contrast */
[data-testid="stSidebar"] {
    background-color: #f1f5f9 !important;
    border-right: 1px solid #cbd5e1 !important;
}
[data-testid="stSidebar"] * {
    color: #0f172a !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #0f172a !important;
    font-weight: 600 !important;
}
/* Radio Navigation Item styling */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 0.5rem 0.75rem !important;
    margin-bottom: 0.35rem !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: #e0e7ff !important;
    border-color: #93c5fd !important;
    transform: translateX(2px) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background-color: #dbeafe !important;
    border-color: #2563eb !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label * {
    color: #0f172a !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}
.sidebar-brand {
    padding: 0.5rem 0 0.85rem;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1rem;
}
.sidebar-brand h2 {
    font-size: 1.35rem;
    font-weight: 800;
    margin: 0;
    color: #1e3a8a !important;
}
.sidebar-brand span {
    font-size: 0.82rem;
    color: #475569 !important;
    font-weight: 600 !important;
}

/* Buttons & Interactive Controls */
.stButton>button, .stDownloadButton>button {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    transition: all 0.2s ease-in-out !important;
}
.stButton>button * {
    color: #0f172a !important;
    font-weight: 700 !important;
}
.stButton>button:hover {
    background-color: #eff6ff !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
}
.stButton>button:hover * {
    color: #1d4ed8 !important;
}

/* Primary Form Submit Button */
.stFormSubmitButton>button,
.stFormSubmitButton>button[kind="primary"],
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 60%, #38bdf8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.12rem !important;
    font-weight: 800 !important;
    padding: 0.85rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
}
.stFormSubmitButton>button *,
.stFormSubmitButton>button[kind="primary"] *,
.stButton>button[kind="primary"] * {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.12rem !important;
}
.stFormSubmitButton>button:hover,
.stButton>button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 60%, #0284c7 100%) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* Form Container Card */
div[data-testid="stForm"] {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 16px !important;
    padding: 1.75rem 2rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
}

/* ALL Input Labels across themes */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] label p,
div[data-testid="stWidgetLabel"] p,
div[data-testid="stWidgetLabel"] span,
label[data-testid="stWidgetLabel"] p {
    color: #0f172a !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    line-height: 1.4 !important;
    opacity: 1 !important;
}

/* Input boxes & number input styling */
div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
    transition: all 0.15s ease-in-out !important;
}
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0.55rem 0.85rem !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
}
div[data-testid="stNumberInput"] button {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
}
div[data-testid="stNumberInput"] button:hover {
    background-color: #e2e8f0 !important;
    color: #1e3a8a !important;
}
div[data-testid="stNumberInput"] button svg {
    fill: #0f172a !important;
}

/* Sliders */
div[data-testid="stSlider"] label p {
    color: #0f172a !important;
    font-weight: 700 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #475569 !important;
    font-weight: 600 !important;
}
div[data-baseweb="slider"] div {
    color: #0f172a !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #2563eb !important;
    border-color: #1d4ed8 !important;
}
[data-testid="stTooltipHoverTarget"] svg {
    fill: #475569 !important;
}

/* Section Header Badges inside form */
.form-section-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.15rem;
    font-weight: 800;
    color: #1e3a8a;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}
.variable-helper {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 500;
    margin-top: -0.25rem;
    margin-bottom: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper Functions for Data, Artifacts & Formatting
# ---------------------------------------------------------------------------
def load_local_data():
    """Load local grocery dataset if present."""
    if DATA_PATH.exists():
        try:
            return pd.read_csv(DATA_PATH)
        except Exception:
            return None
    return None


def read_json_output(filename):
    """Load an artifact JSON safely from output directory."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_feature_title(col_name):
    """Return clean human-readable title for a canonical feature."""
    return FEATURE_LABELS.get(col_name, col_name.replace("_", " ").title())


# Safely attempt to load model and metadata once per run
try:
    _MODEL, META = load_artifact()
    MODEL_STATUS = "ready"
    MODEL_ERROR = None
except ModelArtifactError as exc:
    _MODEL, META = None, None
    MODEL_STATUS = "unavailable"
    MODEL_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Sidebar Navigation & System Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
    <div class="sidebar-brand">
        <h2>🛒 GrocerySpend AI</h2>
        <span style="font-size:0.8rem; color:#64748b; font-weight:500;">Household Expenditure Intelligence</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    PAGES = [
        "📊 Executive Dashboard",
        "🎯 Predict Monthly Spend",
    ]

    selected_page = st.radio("Navigation Menu", PAGES, label_visibility="collapsed")

    st.divider()

    # System status in sidebar
    st.markdown("##### ⚙️ System Status")
    if MODEL_STATUS == "ready" and META:
        st.markdown(
            f"""
        <div style="font-size:0.85rem; background:white; padding:0.75rem; border-radius:8px; border:1px solid #cbd5e1;">
            <b>Model:</b> {META.get('selected_model')}<br>
            <b>Data Kind:</b> <span style="text-transform:uppercase; font-weight:700; color:{'#b45309' if META.get('data_kind')=='synthetic' else '#15803d'}">{META.get('data_kind')}</span><br>
            <b>Status:</b> <span style="color:#15803d; font-weight:bold;">● Active</span><br>
            <b>Features:</b> {len(META.get('features', []))}<br>
            <b>Trained Records:</b> {META.get('training_records', 0):,}
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.error(f"Model offline: {MODEL_ERROR or 'No model metadata'}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("GrocerySpend AI v1.0 • Academic Demonstration")


# ---------------------------------------------------------------------------
# SECTION 1: DASHBOARD
# ---------------------------------------------------------------------------
if selected_page == "📊 Executive Dashboard":
    selected_model_name = META.get("selected_model", "Linear Regression") if META else "Not Trained"

    # Hero Banner
    st.markdown(
        f"""
    <div class="hero-container">
        <div class="hero-title">Household Monthly Grocery Expenditure Intelligence</div>
        <div class="hero-subtitle">
            An end-to-end Machine Learning intelligence dashboard predicting household grocery expenditures in Indian Rupees (₹),
            benchmarking regression models with leakage-free cross-validation, and analyzing key household spending drivers.
        </div>
        <div class="hero-badges">
            <span class="badge-pill badge-model">Selected Regressor: {selected_model_name}</span>
            <span class="badge-pill badge-model">Evaluation Currency: INR (₹)</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load performance metrics
    metrics_data = read_json_output("evaluation_metrics.json")
    test_metrics = metrics_data.get("test_metrics", {}).get(selected_model_name, {}) if metrics_data else {}
    cv_metrics = metrics_data.get("cross_validation", {}).get(selected_model_name, {}) if metrics_data else {}

    # High-level Metric KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        total_rows = META.get("quality_report", {}).get("rows") if META else None
        st.metric("Total Records", f"{total_rows:,}" if total_rows else "—")
    with col2:
        st.metric("Selected Model", selected_model_name)
    with col3:
        cv_rmse = cv_metrics.get("RMSE_mean")
        st.metric("CV RMSE (Mean)", inr(cv_rmse) if cv_rmse is not None else "—")
    with col4:
        test_mae = test_metrics.get("MAE")
        st.metric("Test MAE", inr(test_mae) if test_mae is not None else "—")
    with col5:
        test_r2 = test_metrics.get("R2")
        st.metric("Test R² Score", f"{test_r2:.3f}" if test_r2 is not None else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column overview
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown(
            """
        <div class="custom-card">
            <h3>📖 System Architecture & ML Workflow</h3>
            <p style="color:#475569; font-size:0.92rem; line-height:1.6;">
                GrocerySpend AI provides an end-to-end framework tailored for continuous expenditure prediction:
            </p>
            <ul style="color:#334155; font-size:0.9rem; line-height:1.7;">
                <li><b>Data Ingestion:</b> Supports seeded synthetic generation or user-uploaded authorized household CSVs.</li>
                <li><b>Leakage Prevention:</b> Median imputation and feature standard scaling are fitted exclusively within training folds.</li>
                <li><b>Cross-Validation Model Selection:</b> Selects the best candidate model by mean CV RMSE (tie-broken by MAE) before holdout test evaluation.</li>
                <li><b>Integrity Assurance:</b> Models are stored with cryptographic SHA-256 signatures to detect tampering or corruption.</li>
                <li><b>Indian Context:</b> Currencies and demographic scales reflect realistic household spending patterns in INR.</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🚀 Quick Navigation")
        st.info("🎯 **Predict Monthly Spend**\nSwitch to the **Predict Monthly Spend** section in the sidebar to configure household parameters, select presets, and calculate expected monthly grocery expenditure.")

    with right_col:
        st.markdown(
            f"""
        <div class="custom-card">
            <h3>📋 Active Dataset Provenance</h3>
            <div style="font-size:0.9rem; color:#334155; line-height:1.6;">
                <b>Origin:</b> {META.get('provenance', 'No model loaded') if META else 'Unavailable'}<br>
                <b>Source File:</b> <code>{META.get('source_name', 'grocery_data.csv') if META else 'None'}</code><br>
                <b>Dataset Fingerprint (SHA-256):</b><br>
                <code style="font-size:0.75rem; word-break:break-all;">{META.get('dataset_fingerprint_sha256', '—') if META else '—'}</code><br><br>
                <b>Splitting Strategy:</b> {META.get('split_strategy', 'Standard 80/20 train/test') if META else '—'}<br>
                <b>Training Records:</b> {META.get('training_records', 0):,} rows<br>
                <b>Holdout Test Records:</b> {META.get('test_records', 0):,} rows<br>
                <b>Last Retrained (UTC):</b> {META.get('trained_at_utc', '—')[:19] if META else '—'}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Quick summary of candidate models if metrics available
        if metrics_data and "test_metrics" in metrics_data:
            st.markdown("### 🏆 Candidate Benchmark Summary")
            rows = []
            for m_name, m_vals in metrics_data["test_metrics"].items():
                cv_score = metrics_data.get("cross_validation", {}).get(m_name, {}).get("RMSE_mean", np.nan)
                rows.append({
                    "Model": m_name,
                    "CV RMSE": inr(cv_score) if not np.isnan(cv_score) else "—",
                    "Test MAE": inr(m_vals.get("MAE", 0)),
                    "Test RMSE": inr(m_vals.get("RMSE", 0)),
                    "Test R²": f"{m_vals.get('R2', 0):.3f}",
                    "Selected": "✓ Winner" if m_name == selected_model_name else "",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ---------------------------------------------------------------------------
# SECTION 2: PREDICT MONTHLY GROCERY EXPENDITURE
# ---------------------------------------------------------------------------
elif selected_page == "🎯 Predict Monthly Spend":
    st.markdown(
        """
    <div class="hero-container" style="padding:1.35rem 1.75rem; margin-bottom:1.2rem;">
        <div class="hero-title" style="font-size:1.75rem;">Household Monthly Grocery Spend Estimator</div>
        <div class="hero-subtitle">
            Configure household demographics, shopping frequency, basket preferences, and spending history to calculate
            an intelligent expenditure prediction in Indian Rupees (₹).
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not META or MODEL_STATUS != "ready":
        st.error(f"Cannot generate predictions: {MODEL_ERROR or 'Model artifact not available. Please train a model first.'}")
    else:
        features = META["features"]
        ranges = META.get("feature_ranges", {})
        defaults = META.get("feature_defaults", {})

        # -------------------------------------------------------------------
        # 1. Quick-Fill Household Presets with Rich Metadata
        # -------------------------------------------------------------------
        st.markdown("### ⚡ Quick-Fill Household Presets")
        st.caption("Select a representative household persona to automatically populate realistic shopping and demographic variables:")

        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

        presets = {
            "Mid-Size Family": {
                "icon": "👨‍👩‍👧‍👦",
                "title": "Urban Family of 4",
                "sub": "4 Members • ₹75k Income • 10 Trips",
                "values": {
                    "household_size": 4,
                    "monthly_household_income_inr": 75000.0,
                    "shopping_trips_per_month": 10,
                    "avg_items_per_trip": 26,
                    "branded_spending_pct": 50.0,
                    "fresh_produce_spending_pct": 55.0,
                    "online_shopping_pct": 40.0,
                    "previous_month_expenditure_inr": 9200.0,
                },
            },
            "Young Working Couple": {
                "icon": "👫",
                "title": "Young Working Couple",
                "sub": "2 Members • ₹45k Income • 6 Trips",
                "values": {
                    "household_size": 2,
                    "monthly_household_income_inr": 45000.0,
                    "shopping_trips_per_month": 6,
                    "avg_items_per_trip": 15,
                    "branded_spending_pct": 35.0,
                    "fresh_produce_spending_pct": 65.0,
                    "online_shopping_pct": 25.0,
                    "previous_month_expenditure_inr": 5200.0,
                },
            },
            "Joint Family": {
                "icon": "👨‍👩‍👦‍👦",
                "title": "Large Joint Family",
                "sub": "6+ Members • ₹1.1L Income • 14 Trips",
                "values": {
                    "household_size": 6,
                    "monthly_household_income_inr": 110000.0,
                    "shopping_trips_per_month": 14,
                    "avg_items_per_trip": 40,
                    "branded_spending_pct": 40.0,
                    "fresh_produce_spending_pct": 70.0,
                    "online_shopping_pct": 15.0,
                    "previous_month_expenditure_inr": 14500.0,
                },
            },
            "Online Tech Shopper": {
                "icon": "📱",
                "title": "Quick-Commerce Shopper",
                "sub": "3 Members • ₹1.85L Income • 80% Online",
                "values": {
                    "household_size": 3,
                    "monthly_household_income_inr": 185000.0,
                    "shopping_trips_per_month": 12,
                    "avg_items_per_trip": 30,
                    "branded_spending_pct": 75.0,
                    "fresh_produce_spending_pct": 60.0,
                    "online_shopping_pct": 80.0,
                    "previous_month_expenditure_inr": 15800.0,
                },
            },
        }

        cols_list = [preset_col1, preset_col2, preset_col3, preset_col4]
        for idx, (p_id, p_info) in enumerate(presets.items()):
            with cols_list[idx]:
                st.markdown(
                    f"""
                <div style="background:#ffffff; border:1.5px solid #cbd5e1; border-radius:12px; padding:0.75rem 0.85rem; text-align:center; min-height:85px; margin-bottom:0.4rem;">
                    <div style="font-size:1.35rem; margin-bottom:0.15rem;">{p_info['icon']}</div>
                    <div style="font-weight:700; color:#0f172a; font-size:0.88rem;">{p_info['title']}</div>
                    <div style="font-size:0.75rem; color:#64748b; font-weight:500;">{p_info['sub']}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Load {p_id}", key=f"btn_preset_{idx}", use_container_width=True):
                    for k, v in p_info["values"].items():
                        if k in features:
                            st.session_state[f"inp_{k}"] = v
                    st.rerun()

        # -------------------------------------------------------------------
        # 2. Interactive Input Form with Meaningful Variables
        # -------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)

        input_values = {}
        with st.form("prediction_form", clear_on_submit=False):
            st.markdown(
                """
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; border-bottom:2px solid #e2e8f0; padding-bottom:0.75rem;">
                <div>
                    <h3 style="margin:0; color:#0f172a; font-size:1.35rem; font-weight:800;">📝 Household Spending Parameter Inputs</h3>
                    <span style="font-size:0.85rem; color:#475569; font-weight:500;">Adjust each meaningful variable to reflect your household profile and shopping habits.</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            left_col, right_col = st.columns(2)

            with left_col:
                # ---------------- SECTION A: DEMOGRAPHICS & INCOME ----------------
                st.markdown('<div class="form-section-title">👥 Household Demographics & Income</div>', unsafe_allow_html=True)

                if "household_size" in features:
                    def_size = int(st.session_state.get("inp_household_size", defaults.get("household_size", 3.0)))
                    st.markdown('<div class="variable-helper">Number of family members permanently living and sharing meals in the home.</div>', unsafe_allow_html=True)
                    input_values["household_size"] = st.number_input(
                        "👥 Household Size (Family Members)",
                        min_value=1,
                        max_value=12,
                        value=def_size,
                        step=1,
                        help="Number of people sharing grocery purchases and meals daily.",
                    )

                if "monthly_household_income_inr" in features:
                    def_income = float(st.session_state.get("inp_monthly_household_income_inr", defaults.get("monthly_household_income_inr", 55000.0)))
                    st.markdown('<div class="variable-helper">Total combined take-home monthly earnings across all earning members.</div>', unsafe_allow_html=True)
                    input_values["monthly_household_income_inr"] = st.number_input(
                        "💰 Monthly Household Income (Take-Home ₹ INR)",
                        min_value=5000.0,
                        max_value=1000000.0,
                        value=def_income,
                        step=2500.0,
                        help="Net take-home monthly income in Indian Rupees.",
                    )

                # ---------------- SECTION B: SHOPPING HABITS ----------------
                st.markdown('<div class="form-section-title" style="margin-top:1.5rem;">🛒 Grocery Shopping Frequency & Volume</div>', unsafe_allow_html=True)

                if "shopping_trips_per_month" in features:
                    def_trips = int(st.session_state.get("inp_shopping_trips_per_month", defaults.get("shopping_trips_per_month", 9.0)))
                    st.markdown('<div class="variable-helper">Combined in-store market visits and digital delivery app orders placed per month.</div>', unsafe_allow_html=True)
                    input_values["shopping_trips_per_month"] = st.slider(
                        "🏬 Monthly Shopping Frequency (Trips & App Orders)",
                        min_value=1,
                        max_value=31,
                        value=def_trips,
                        help="Total grocery ordering and market trips made per month.",
                    )

                if "avg_items_per_trip" in features:
                    def_items = int(st.session_state.get("inp_avg_items_per_trip", defaults.get("avg_items_per_trip", 22.0)))
                    st.markdown('<div class="variable-helper">Typical count of distinct grocery products purchased during each shopping trip.</div>', unsafe_allow_html=True)
                    input_values["avg_items_per_trip"] = st.slider(
                        "🛍️ Average Basket Size (Items Purchased per Trip)",
                        min_value=1,
                        max_value=80,
                        value=def_items,
                        help="Average number of unique grocery items per checkout basket.",
                    )

            with right_col:
                # ---------------- SECTION C: BASKET COMPOSITION ----------------
                st.markdown('<div class="form-section-title">🏷️ Basket Composition & Channels</div>', unsafe_allow_html=True)

                if "branded_spending_pct" in features:
                    def_branded = float(st.session_state.get("inp_branded_spending_pct", defaults.get("branded_spending_pct", 42.0)))
                    st.markdown('<div class="variable-helper">Share allocated to packaged FMCG and branded goods vs. unbranded staples.</div>', unsafe_allow_html=True)
                    input_values["branded_spending_pct"] = st.slider(
                        "🏷️ Branded Product Share (% of Total Grocery Spend)",
                        min_value=0.0,
                        max_value=100.0,
                        value=def_branded,
                        step=1.0,
                        help="Proportion spent on packaged branded foods and toiletries.",
                    )

                if "fresh_produce_spending_pct" in features:
                    def_fresh = float(st.session_state.get("inp_fresh_produce_spending_pct", defaults.get("fresh_produce_spending_pct", 58.0)))
                    st.markdown('<div class="variable-helper">Share allocated to vegetables, fresh fruits, milk, dairy, eggs, and bakery items.</div>', unsafe_allow_html=True)
                    input_values["fresh_produce_spending_pct"] = st.slider(
                        "🥬 Fresh Produce & Perishables Share (% of Grocery Spend)",
                        min_value=0.0,
                        max_value=100.0,
                        value=def_fresh,
                        step=1.0,
                        help="Percentage of grocery budget devoted to perishable daily staples.",
                    )

                if "online_shopping_pct" in features:
                    def_online = float(st.session_state.get("inp_online_shopping_pct", defaults.get("online_shopping_pct", 28.0)))
                    st.markdown('<div class="variable-helper">Share of orders placed through quick-commerce apps (Blinkit, Zepto, Instamart, etc.).</div>', unsafe_allow_html=True)
                    input_values["online_shopping_pct"] = st.slider(
                        "📱 Quick-Commerce / Online Share (% of All Grocery Orders)",
                        min_value=0.0,
                        max_value=100.0,
                        value=def_online,
                        step=1.0,
                        help="Percentage of orders delivered via online delivery apps.",
                    )

                # ---------------- SECTION D: SPENDING HISTORY ----------------
                st.markdown('<div class="form-section-title" style="margin-top:1.5rem;">📅 Historical Spending Benchmark</div>', unsafe_allow_html=True)

                if "previous_month_expenditure_inr" in features:
                    def_prior = float(st.session_state.get("inp_previous_month_expenditure_inr", defaults.get("previous_month_expenditure_inr", 7500.0)))
                    st.markdown('<div class="variable-helper">Actual verified grocery expenditure recorded in the immediately preceding month.</div>', unsafe_allow_html=True)
                    input_values["previous_month_expenditure_inr"] = st.number_input(
                        "📅 Previous Month Grocery Spend (Baseline ₹ INR)",
                        min_value=500.0,
                        max_value=250000.0,
                        value=def_prior,
                        step=500.0,
                        help="Known grocery expense from the previous month in INR.",
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮 Calculate Monthly Grocery Expenditure Estimate", use_container_width=True)

        if submitted:
            try:
                prediction_result = predict_expenditure(input_values)
                st.session_state["latest_prediction"] = {"values": input_values, **prediction_result}
            except (ValueError, ModelArtifactError) as exc:
                st.error(f"Prediction Error: {exc}")

        # -------------------------------------------------------------------
        # 3. Rich Interactive Results Dashboard
        # -------------------------------------------------------------------
        pred = st.session_state.get("latest_prediction")
        if pred:
            pred_amount = pred["prediction_inr"]
            income = pred["values"].get("monthly_household_income_inr", 0)
            prior = pred["values"].get("previous_month_expenditure_inr", 0)
            hh_size = pred["values"].get("household_size", 1)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
            <div class="prediction-box">
                <div class="prediction-label">Estimated Monthly Grocery Expenditure</div>
                <div class="prediction-value">{inr(pred_amount)}</div>
                <div class="prediction-subtext">
                    Predicted via <b>{pred['model_name']}</b> • Model Version <b>{pred['model_version']}</b> • Schema: <b>{len(pred['values'])} Variables</b>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # 4-Grid Key Analytical Metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                per_capita = pred_amount / max(1, hh_size)
                st.metric("Per-Person Monthly Spend", inr(per_capita), help="Estimated monthly expenditure divided by family members.")

            with kpi2:
                daily_spend = pred_amount / 30.0
                st.metric("Projected Daily Spend", f"{inr(daily_spend)} / day", help="Average daily grocery budget requirement.")

            with kpi3:
                if income > 0:
                    income_share = (pred_amount / income) * 100
                    st.metric("Share of Monthly Income", f"{income_share:.1f}%", help="Percentage of take-home income dedicated to groceries.")
                else:
                    st.metric("Share of Monthly Income", "—")

            with kpi4:
                if prior > 0:
                    diff = pred_amount - prior
                    pct_change = (diff / prior) * 100
                    st.metric(
                        "Month-over-Month Delta",
                        inr(diff),
                        delta=f"{pct_change:+.1f}% vs. Last Month",
                        delta_color="inverse",
                    )
                else:
                    st.metric("Month-over-Month Delta", "—")

            # Out of training range caution
            if pred["out_of_training_range"]:
                st.warning(
                    f"⚠️ **Extrapolation Notice:** The following inputs fall outside observed training ranges: "
                    f"**{', '.join(get_feature_title(f) for f in pred['out_of_training_range'])}**. "
                    "Predictions may have wider error margins when inputs exceed training bounds."
                )

            # Interactive Plotly Analytics Section
            st.markdown("### 📊 Interactive Spending Breakdown & Analytics")

            chart_c1, chart_c2 = st.columns(2)

            with chart_c1:
                # 1. Spend Allocation Donut Chart
                fresh_pct = pred["values"].get("fresh_produce_spending_pct", 50.0)
                branded_pct = pred["values"].get("branded_spending_pct", 30.0)
                staples_pct = max(0.0, 100.0 - (fresh_pct * 0.5 + branded_pct * 0.5))

                fresh_inr = pred_amount * (fresh_pct / 100.0) * 0.5
                branded_inr = pred_amount * (branded_pct / 100.0) * 0.5
                staples_inr = max(0.0, pred_amount - fresh_inr - branded_inr)

                donut_labels = ["🥬 Fresh Produce & Dairy", "🏷️ Packaged Branded FMCG", "🍚 Staples & Essentials"]
                donut_values = [fresh_inr, branded_inr, staples_inr]

                fig_donut = go.Figure(data=[
                    go.Pie(
                        labels=donut_labels,
                        values=donut_values,
                        hole=0.55,
                        marker_colors=["#10b981", "#3b82f6", "#f59e0b"],
                        textinfo="label+percent",
                        hovertemplate="<b>%{label}</b><br>Estimated Amount: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
                    )
                ])
                fig_donut.update_layout(
                    title="Estimated Monthly Basket Category Allocation",
                    height=340,
                    margin=dict(l=20, r=20, t=45, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with chart_c2:
                # 2. Month-over-Month Comparison
                if prior > 0:
                    fig_comp = go.Figure(data=[
                        go.Bar(
                            name="Expenditure",
                            x=["Previous Month", "Predicted This Month"],
                            y=[prior, pred_amount],
                            marker_color=["#94a3b8", "#2563eb"],
                            text=[inr(prior), inr(pred_amount)],
                            textposition="auto",
                            hovertemplate="<b>%{x}</b><br>Spend: ₹%{y:,.0f}<extra></extra>",
                        )
                    ])
                    diff_amt = pred_amount - prior
                    trend_txt = f"{'+' if diff_amt >= 0 else ''}{inr(diff_amt)} ({diff_amt/prior*100:+.1f}%)"
                    fig_comp.update_layout(
                        title=f"MoM Comparison (Trend: {trend_txt})",
                        yaxis_title="Amount (₹ INR)",
                        height=340,
                        margin=dict(l=20, r=20, t=45, b=20),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

            # Income Budget Health Assessment
            if income > 0:
                income_pct = (pred_amount / income) * 100
                if income_pct < 15.0:
                    st.success(
                        f"🟢 **Budget Health: Economical Allocation ({income_pct:.1f}% of income)**. "
                        f"Your estimated grocery expenditure represents a conservative and healthy share of your monthly take-home earnings."
                    )
                elif income_pct <= 25.0:
                    st.info(
                        f"🟡 **Budget Health: Balanced Allocation ({income_pct:.1f}% of income)**. "
                        f"Your estimated grocery expenditure aligns with typical urban Indian middle-income household consumption patterns."
                    )
                else:
                    st.warning(
                        f"🟠 **Budget Health: High Priority Expense ({income_pct:.1f}% of income)**. "
                        f"Grocery spending represents a substantial portion of monthly earnings, in line with Engel's Law for essential food consumption."
                    )

            # Detailed Profile Summary & Export
            with st.expander("📄 View Submitted Parameter Values & Download Report", expanded=False):
                submitted_df = pd.DataFrame([pred["values"]]).rename(columns={c: get_feature_title(c) for c in pred["values"]})
                st.dataframe(submitted_df, use_container_width=True)

                export_data = {
                    **pred["values"],
                    "predicted_monthly_expenditure_inr": round(pred_amount, 2),
                    "per_person_expenditure_inr": round(per_capita, 2),
                    "daily_grocery_expenditure_inr": round(daily_spend, 2),
                    "share_of_income_pct": round(income_share, 2) if income > 0 else 0,
                    "model_used": pred["model_name"],
                    "data_kind": pred["data_kind"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                csv_bytes = pd.DataFrame([export_data]).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Prediction Report (CSV)",
                    data=csv_bytes,
                    file_name="groceryspend_prediction_report.csv",
                    mime="text/csv",
                )
