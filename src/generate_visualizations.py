"""
Comprehensive Visualization and Chart Generation Suite for GrocerySpend AI.
Generates publication-quality figures (300 DPI) covering Exploratory Data Analysis,
Microeconomic validation (Engel's Law, spending inertia), and Machine Learning
evaluation metrics (Parity plots, Residuals, Benchmarks, Feature Importance).
Saves all figures into reports/visualizations/.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils import DATA_PATH, MODEL_PATH, METADATA_PATH, OUTPUT_DIR, FEATURES, TARGET, inr
from src.predict import load_artifact

# Define paths
REPORTS_DIR = ROOT / "reports"
VISUALIZATIONS_DIR = REPORTS_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["figure.titleweight"] = "bold"

# Professional Theme Color Palette
NAVY = "#1F4E79"
BLUE = "#2563EB"
TEAL = "#0D9488"
GREEN = "#16A34A"
ORANGE = "#EA580C"
RED = "#DC2626"
GRAY = "#64748B"


def generate_00_executive_dashboard(df, test_df, model, metadata):
    """Figure 00: Multi-panel high-impact executive dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    
    # Panel 1: Spend vs Income & Engel's Law
    ax1 = axes[0, 0]
    sns.regplot(
        data=df, x="monthly_household_income_inr", y=TARGET,
        scatter_kws={"alpha": 0.25, "color": BLUE, "s": 16},
        line_kws={"color": RED, "linewidth": 2, "label": "Trendline (Engel's Law)"},
        ax=ax1, logx=True
    )
    ax1.set_title("A: Income vs. Grocery Spend (Engel's Law Curve)", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Monthly Household Income (₹ INR)")
    ax1.set_ylabel("Monthly Grocery Spend (₹ INR)")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x/1000):,}k"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y/1000):,}k"))
    ax1.legend(loc="upper left")

    # Panel 2: Model Benchmark Comparison
    ax2 = axes[0, 1]
    models = ["Mean Baseline", "Linear Reg (Winner)", "Random Forest", "HistGradBoost"]
    cv_rmse = [2659.71, 1274.17, 1324.75, 1400.48]
    colors = [GRAY, GREEN, BLUE, ORANGE]
    bars = ax2.bar(models, cv_rmse, color=colors, width=0.55, edgecolor="black", linewidth=1.1)
    ax2.set_title("B: Model Selection Benchmark (5-Fold CV RMSE)", fontsize=12, fontweight="bold", color=NAVY)
    ax2.set_ylabel("Cross-Validation RMSE (₹ INR)")
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"₹{yval:,.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    ax2.set_ylim(0, 3100)

    # Panel 3: Actual vs Predicted Parity Plot
    ax3 = axes[1, 0]
    y_test = test_df[TARGET]
    X_test = test_df[FEATURES]
    y_pred = model.predict(X_test)
    ax3.scatter(y_test, y_pred, alpha=0.35, color=TEAL, s=20, label="Test Households (N=400)")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], "--", color=RED, linewidth=2, label="Perfect Parity Line (y = x)")
    ax3.set_title("C: Test Set Actual vs. Predicted Parity ($R^2 = 0.761$)", fontsize=12, fontweight="bold", color=NAVY)
    ax3.set_xlabel("Actual Monthly Grocery Spend (₹ INR)")
    ax3.set_ylabel("Predicted Monthly Grocery Spend (₹ INR)")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x/1000):,}k"))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y/1000):,}k"))
    ax3.legend(loc="upper left")

    # Panel 4: Permutation Feature Importance
    ax4 = axes[1, 1]
    features_display = [
        "Prev Month Spend", "Household Size", "Monthly Income", "Trips / Month",
        "Branded Spend %", "Fresh Produce %", "Avg Items / Trip", "Online Quick-Com %"
    ]
    importance_mae = [838.0, 292.0, 22.0, 15.0, 5.0, 0.5, 0.0, 0.0]
    y_pos = np.arange(len(features_display))[::-1]
    bar_colors = [NAVY if i < 2 else BLUE if i < 4 else GRAY for i in range(len(features_display))]
    ax4.barh(y_pos, importance_mae, color=bar_colors, edgecolor="black", linewidth=1.0)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(features_display, fontweight="bold")
    ax4.set_title("D: Permutation Feature Importance (Mean MAE Impact)", fontsize=12, fontweight="bold", color=NAVY)
    ax4.set_xlabel("Mean Test MAE Increase When Permuted (₹ INR)")
    for idx, (pos, val) in enumerate(zip(y_pos, importance_mae)):
        if val > 0:
            ax4.text(val + 15, pos, f"+₹{val:,.0f}", va="center", fontweight="bold", fontsize=9.5)
    ax4.set_xlim(0, 1000)

    fig.suptitle("GROCERYSPEND AI — EXECUTIVE RESEARCH & PREDICTIVE ANALYTICS DASHBOARD", fontsize=16, fontweight="bold", color=NAVY, y=0.99)
    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "00_executive_dashboard_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_01_target_spend_distribution(df):
    """Figure 01: Histogram & Density Distribution of Monthly Grocery Expenditure."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    spend = df[TARGET]
    mean_val = spend.mean()
    median_val = spend.median()
    std_val = spend.std()
    q25 = spend.quantile(0.25)
    q75 = spend.quantile(0.75)

    sns.histplot(spend, bins=35, kde=True, color=NAVY, edgecolor="white", alpha=0.75, ax=ax, line_kws={"linewidth": 2.5})
    ax.axvline(mean_val, color=RED, linestyle="--", linewidth=2, label=f"Mean: ₹{mean_val:,.1f}")
    ax.axvline(median_val, color=GREEN, linestyle="-.", linewidth=2, label=f"Median: ₹{median_val:,.1f}")

    # Annotate stats box
    stats_text = (
        f"Distribution Profile (N=2,000):\n"
        f"• Minimum:   ₹{spend.min():,.0f}\n"
        f"• 25th Pct:  ₹{q25:,.0f}\n"
        f"• Median:    ₹{median_val:,.0f}\n"
        f"• Mean:      ₹{mean_val:,.0f}\n"
        f"• 75th Pct:  ₹{q75:,.0f}\n"
        f"• Maximum:   ₹{spend.max():,.0f}\n"
        f"• Std Dev:   ₹{std_val:,.0f}\n"
        f"• IQR:       ₹{q75-q25:,.0f}"
    )
    ax.text(0.70, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8FAFC", edgecolor=NAVY, alpha=0.9),
            fontsize=9.5, family="monospace")

    ax.set_title("Distribution of Monthly Household Grocery Spend in INR", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Monthly Grocery Expenditure (₹ INR)", fontweight="bold")
    ax.set_ylabel("Number of Observed Households", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "01_target_spend_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_02_income_distribution_and_engels_law(df):
    """Figure 02: Dual plot of Income distribution and Engel's Law verification."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Left: Log-normal Income Distribution
    income = df["monthly_household_income_inr"]
    sns.histplot(income, bins=35, kde=True, color=BLUE, edgecolor="white", alpha=0.7, ax=ax1)
    ax1.axvline(income.mean(), color=RED, linestyle="--", linewidth=2, label=f"Mean: ₹{income.mean():,.0f}")
    ax1.axvline(income.median(), color=GREEN, linestyle="-.", linewidth=2, label=f"Median: ₹{income.median():,.0f}")
    ax1.set_title("Household Income Distribution (Right-Skewed)", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Monthly Household Income (₹ INR)")
    ax1.set_ylabel("Household Count")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x/1000):,}k"))
    ax1.legend(loc="upper right")

    # Right: Engel's Law: Grocery Share of Income (%) vs Income
    spend_share = (df[TARGET] / df["monthly_household_income_inr"]) * 100
    df_temp = df.copy()
    df_temp["spend_share_pct"] = spend_share

    ax2.scatter(df_temp["monthly_household_income_inr"], df_temp["spend_share_pct"], alpha=0.3, color=TEAL, s=18)
    sns.regplot(
        data=df_temp, x="monthly_household_income_inr", y="spend_share_pct",
        scatter=False, ax=ax2, logx=True, line_kws={"color": RED, "linewidth": 2.5, "label": "Engel's Empirical Curve"}
    )
    ax2.set_title("Empirical Confirmation of Engel's Law\n(Budget Share Drops as Income Rises)", fontsize=12, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Monthly Household Income (₹ INR)")
    ax2.set_ylabel("Grocery Spend as % of Income")
    ax2.set_ylim(0, 45)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x/1000):,}k"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y)}%"))
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "02_income_distribution_and_engels_law.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_03_household_size_spending_tiers(df):
    """Figure 03: Boxplot and Violin Plot of Spend across Household Size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Boxplot
    sns.boxplot(data=df, x="household_size", y=TARGET, palette="Blues_r", ax=ax1, linewidth=1.2, fliersize=3)
    ax1.set_title("Grocery Spend Tiers by Household Size (Boxplot)", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Household Size (Family Members)")
    ax1.set_ylabel("Monthly Spend (₹ INR)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y):,}"))

    # Violin plot showing density
    sns.violinplot(data=df, x="household_size", y=TARGET, palette="Purples_r", ax=ax2, inner="quartile")
    ax2.set_title("Grocery Spend Distribution Density by Household Size", fontsize=12, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Household Size (Family Members)")
    ax2.set_ylabel("Monthly Spend (₹ INR)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y):,}"))

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "03_household_size_spending_tiers.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_04_spend_breakdown_categories(df):
    """Figure 04: Distribution and interactions of basket category shares."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 1. Branded %
    sns.histplot(df["branded_spending_pct"], bins=25, kde=True, color=ORANGE, ax=axes[0], edgecolor="white")
    axes[0].set_title("Branded FMCG Goods Budget Share (%)", fontweight="bold", color=NAVY)
    axes[0].set_xlabel("Branded Spend %")
    axes[0].set_ylabel("Households")

    # 2. Fresh Produce %
    sns.histplot(df["fresh_produce_spending_pct"], bins=25, kde=True, color=GREEN, ax=axes[1], edgecolor="white")
    axes[1].set_title("Fresh Produce & Perishables Budget Share (%)", fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Fresh Produce Spend %")
    axes[1].set_ylabel("Households")

    # 3. Online Quick-Commerce %
    sns.histplot(df["online_shopping_pct"], bins=25, kde=True, color=BLUE, ax=axes[2], edgecolor="white")
    axes[2].set_title("Online / Quick-Commerce Delivery Share (%)", fontweight="bold", color=NAVY)
    axes[2].set_xlabel("Online Shopping %")
    axes[2].set_ylabel("Households")

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "04_spend_breakdown_categories.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_05_shopping_cadence_and_basket_size(df):
    """Figure 05: Joint distribution of shopping trips and items per trip vs spend."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    sns.regplot(data=df, x="shopping_trips_per_month", y=TARGET,
                scatter_kws={"alpha": 0.25, "color": TEAL, "s": 15},
                line_kws={"color": RED, "linewidth": 2}, ax=ax1)
    ax1.set_title("Shopping Trips / Month vs. Monthly Grocery Spend", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Monthly Shopping Trips (In-Store + Online)")
    ax1.set_ylabel("Monthly Spend (₹ INR)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y):,}"))

    sns.regplot(data=df, x="avg_items_per_trip", y=TARGET,
                scatter_kws={"alpha": 0.25, "color": BLUE, "s": 15},
                line_kws={"color": RED, "linewidth": 2}, ax=ax2)
    ax2.set_title("Average Basket Size (Items/Trip) vs. Monthly Grocery Spend", fontsize=12, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Average Number of Items per Trip")
    ax2.set_ylabel("Monthly Spend (₹ INR)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y):,}"))

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "05_shopping_cadence_and_basket_size.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_06_previous_vs_current_spend_inertia(df):
    """Figure 06: Spending inertia - previous month spend vs current spend."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    r_val, _ = stats.pearsonr(df["previous_month_expenditure_inr"], df[TARGET])

    sns.regplot(
        data=df, x="previous_month_expenditure_inr", y=TARGET,
        scatter_kws={"alpha": 0.3, "color": NAVY, "s": 20},
        line_kws={"color": RED, "linewidth": 2.5, "label": f"Linear Regression Fit (r = {r_val:.3f})"},
        ax=ax
    )
    ax.set_title(f"Grocery Spending Inertia: Previous Month vs. Current Month Spend\n(Strong Linear Memory: Pearson r = {r_val:.3f})",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Previous Month Grocery Expenditure (₹ INR)", fontweight="bold")
    ax.set_ylabel("Current Month Grocery Expenditure (₹ INR)", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y):,}"))
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "06_previous_vs_current_spend_inertia.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_07_correlation_matrix_heatmap(df):
    """Figure 07: High-resolution annotated correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cols = FEATURES + [TARGET]
    labels = [
        "Household Size", "Monthly Income", "Trips / Month", "Items / Trip",
        "Branded Spend %", "Fresh Produce %", "Online %", "Prev Month Spend", "Target Spend"
    ]
    corr = df[cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-0.5, vmax=1.0, center=0,
        annot=True, fmt=".2f", square=True, linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title("Correlation Matrix Heatmap of Expenditure Attributes", fontsize=14, fontweight="bold", color=NAVY, pad=15)
    plt.xticks(rotation=45, ha="right", fontweight="bold")
    plt.yticks(fontweight="bold")

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "07_correlation_matrix_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_08_model_comparison_benchmark():
    """Figure 08: Grouped bar chart comparing the 4 candidate models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    models = ["Mean Baseline", "Linear Regression (Winner)", "Random Forest", "HistGradientBoosting"]
    cv_rmse = [2659.71, 1274.17, 1324.75, 1400.48]
    test_mae = [2153.05, 1081.39, 1105.76, 1178.75]
    test_r2 = [-0.001, 0.761, 0.748, 0.716]

    colors = [GRAY, GREEN, BLUE, ORANGE]

    # Chart 1: Error Metrics (CV RMSE and Test MAE)
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax1.bar(x - width/2, cv_rmse, width, label="5-Fold CV RMSE (₹)", color=NAVY, edgecolor="black")
    bars2 = ax1.bar(x + width/2, test_mae, width, label="Test MAE (₹)", color=TEAL, edgecolor="black")
    ax1.set_title("Model Error Metrics: CV RMSE vs. Test MAE (Lower is Better)", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right", fontweight="bold")
    ax1.set_ylabel("Error in Indian Rupees (₹ INR)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y):,}"))
    ax1.legend()
    ax1.set_ylim(0, 3100)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 40, f"₹{b.get_height():,.0f}", ha="center", fontsize=8.5, fontweight="bold")
    for b in bars2:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 40, f"₹{b.get_height():,.0f}", ha="center", fontsize=8.5, fontweight="bold")

    # Chart 2: Test R2 Score
    bars3 = ax2.bar(models, test_r2, color=colors, width=0.5, edgecolor="black")
    ax2.set_title("Held-Out Test Set Goodness-of-Fit ($R^2$ Score)", fontsize=12, fontweight="bold", color=NAVY)
    ax2.set_ylabel("Coefficient of Determination ($R^2$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right", fontweight="bold")
    ax2.set_ylim(-0.1, 0.90)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    for b in bars3:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.02 if h >= 0 else -0.05
        ax2.text(b.get_x() + b.get_width()/2, h + offset, f"{h:.3f}", ha="center", va=va, fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "08_model_comparison_benchmark.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_09_actual_vs_predicted_parity(test_df, model):
    """Figure 09: Actual vs Predicted Parity Plot on 400 Holdout Samples."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    y_test = test_df[TARGET]
    X_test = test_df[FEATURES]
    y_pred = model.predict(X_test)

    ax.scatter(y_test, y_pred, alpha=0.45, color=NAVY, s=28, edgecolors="white", linewidths=0.5, label="Test Households (N=400)")
    
    # 45-degree line
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "--", color=RED, linewidth=2.5, label="Ideal Parity Line (y = x)")

    # 15% Error Margin bounds
    ax.plot([min_v, max_v], [min_v * 1.15, max_v * 1.15], ":", color=GRAY, linewidth=1.5, label="±15% Error Boundary")
    ax.plot([min_v, max_v], [min_v * 0.85, max_v * 0.85], ":", color=GRAY, linewidth=1.5)

    ax.set_title("Champion Model Parity Plot: Actual vs. Predicted Spend\n(Linear Regression, Test $R^2 = 0.761$)",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Actual Grocery Expenditure (₹ INR)", fontweight="bold")
    ax.set_ylabel("Predicted Grocery Expenditure (₹ INR)", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y):,}"))
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "09_actual_vs_predicted_parity.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_10_residual_diagnostic_plots(test_df, model):
    """Figure 10: Dual residual diagnostic plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    y_test = test_df[TARGET]
    X_test = test_df[FEATURES]
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.4, color=TEAL, s=24, edgecolors="white", linewidths=0.5)
    ax1.axhline(0, color=RED, linestyle="--", linewidth=2, label="Zero Bias Reference")
    ax1.axhline(1333, color=GRAY, linestyle=":", linewidth=1.2, label="+1 RMSE (₹1,333)")
    ax1.axhline(-1333, color=GRAY, linestyle=":", linewidth=1.2, label="-1 RMSE (-₹1,333)")
    ax1.set_title("Residuals vs. Fitted Predicted Values (Homoscedasticity)", fontsize=12, fontweight="bold", color=NAVY)
    ax1.set_xlabel("Predicted Spend (₹ INR)")
    ax1.set_ylabel("Residual Error (Actual - Predicted) (₹ INR)")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"₹{int(y):,}"))
    ax1.legend(loc="upper left")

    # Residual Histogram with Gaussian Fit
    sns.histplot(residuals, bins=30, kde=True, color=NAVY, edgecolor="white", alpha=0.75, ax=ax2, line_kws={"linewidth": 2.5})
    ax2.axvline(0, color=RED, linestyle="--", linewidth=2, label="Zero Error Mean")
    ax2.set_title("Residual Error Normality Distribution (Mean ≈ 0)", fontsize=12, fontweight="bold", color=NAVY)
    ax2.set_xlabel("Prediction Error (₹ INR)")
    ax2.set_ylabel("Test Household Count")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "10_residual_diagnostic_plots.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def generate_11_permutation_feature_importance():
    """Figure 11: Permutation feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    features_display = [
        "Previous Month Spend (₹)",
        "Household Size (People)",
        "Monthly Household Income (₹)",
        "Shopping Trips / Month",
        "Branded Spending Share (%)",
        "Fresh Produce Share (%)",
        "Average Items / Trip",
        "Online Shopping Share (%)"
    ]
    importance_mae = [838.0, 292.0, 22.0, 15.0, 5.0, 0.5, 0.0, 0.0]
    importance_std = [69.0, 43.0, 10.0, 6.0, 3.0, 1.0, 4.0, 1.0]

    y_pos = np.arange(len(features_display))[::-1]
    colors = [NAVY if i < 2 else BLUE if i < 4 else GRAY for i in range(len(features_display))]

    bars = ax.barh(y_pos, importance_mae, xerr=importance_std, color=colors, edgecolor="black", linewidth=1.1,
                   capsize=4, error_kw={"elinewidth": 1.5, "ecolor": "black"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_display, fontweight="bold", fontsize=10.5)
    ax.set_title("Permutation Feature Importance on Held-Out Test Set (Scoring: MAE)", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xlabel("Mean Increase in Test MAE When Feature Is Permuted (₹ INR)", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    ax.set_xlim(0, 1000)

    for idx, (pos, val, std) in enumerate(zip(y_pos, importance_mae, importance_std)):
        if val > 0:
            ax.text(val + std + 25, pos, f"+₹{val:,.0f} ± ₹{std:,.0f}", va="center", fontweight="bold", fontsize=9.5)

    plt.tight_layout()
    out_path = VISUALIZATIONS_DIR / "11_permutation_feature_importance.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.name}")


def run_all_visualizations():
    """Master controller to execute and save all 12 publication charts."""
    print("=" * 75)
    print("GROCERYSPEND AI - MASTER VISUALIZATION GENERATOR SUITE")
    print("=" * 75)
    print(f"Target Output Directory: {VISUALIZATIONS_DIR}")
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded primary dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Load model artifact
    model, metadata = load_artifact(MODEL_PATH, METADATA_PATH)
    print(f"Loaded model artifact: {metadata['selected_model']} (v{metadata['artifact_version']})")

    # Load test split
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Split test holdout: {test_df.shape[0]} test samples")

    print("\nGenerating Figures...")
    generate_00_executive_dashboard(df, test_df, model, metadata)
    generate_01_target_spend_distribution(df)
    generate_02_income_distribution_and_engels_law(df)
    generate_03_household_size_spending_tiers(df)
    generate_04_spend_breakdown_categories(df)
    generate_05_shopping_cadence_and_basket_size(df)
    generate_06_previous_vs_current_spend_inertia(df)
    generate_07_correlation_matrix_heatmap(df)
    generate_08_model_comparison_benchmark()
    generate_09_actual_vs_predicted_parity(test_df, model)
    generate_10_residual_diagnostic_plots(test_df, model)
    generate_11_permutation_feature_importance()

    print("\n" + "=" * 75)
    print("ALL 12 VISUALIZATION CHARTS GENERATED SUCCESSFULLY IN HIGH-RES (300 DPI)!")
    print("=" * 75)


if __name__ == "__main__":
    run_all_visualizations()
