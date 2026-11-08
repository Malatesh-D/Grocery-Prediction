"""Metrics and plot generation for trained regression models."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import OUTPUT_DIR, ensure_directories

def regression_metrics(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred))}

def save_eda(df, features, target):
    ensure_directories()
    sns.set_theme(style="whitegrid")
    df.hist(figsize=(14, 10), bins=25, color="#4f46e5")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150); plt.close()
    plt.figure(figsize=(10, 7)); sns.heatmap(df[features + [target]].corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    numeric_features=[c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    for ax,col in zip(axes,numeric_features[:2]):
        sns.scatterplot(data=df,x=col,y=target,alpha=.45,ax=ax,color="#0f766e")
        ax.set(xlabel=col.replace("_"," ").title(),ylabel="Monthly expenditure (₹)",title=f"{col.replace('_',' ').title()} and expenditure")
    if not numeric_features: axes[0].text(.5,.5,"No numeric features",ha="center")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "size_income_analysis.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.flat, features[:4]):
        sns.regplot(data=df, x=col, y=target, scatter_kws={"alpha": .25, "s": 12}, line_kws={"color": "#dc2626"}, ax=ax)
        ax.set_title(f"{col.replace('_', ' ').title()} vs expenditure")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "feature_target_relationships.png", dpi=150); plt.close(fig)

def save_model_plots(y_true, y_pred, results, feature_importance):
    ensure_directories()
    fig, ax = plt.subplots(figsize=(7, 6)); density=ax.hexbin(y_true,y_pred,gridsize=28,mincnt=1,cmap="Purples"); fig.colorbar(density,ax=ax,label="Test sample count")
    low, high = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred)); ax.plot([low, high], [low, high], "--", color="#dc2626")
    ax.set(xlabel="Actual expenditure (₹)", ylabel="Predicted expenditure (₹)", title="Actual vs predicted"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160); plt.close(fig)
    residual = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5)); density=ax.hexbin(y_pred,residual,gridsize=28,mincnt=1,cmap="GnBu"); fig.colorbar(density,ax=ax,label="Test sample count"); ax.axhline(0, ls="--", color="#dc2626")
    ax.set(xlabel="Predicted expenditure (₹)", ylabel="Residual (actual − predicted) (₹)", title="Residual plot"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "residual_plot.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); sns.histplot(residual, bins=25, kde=True, ax=ax, color="#4f46e5")
    ax.set(xlabel="Prediction error (actual − predicted), ₹", ylabel="Test households", title="Test error distribution")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "error_distribution.png", dpi=160); plt.close(fig)
    pd.DataFrame(results).T.rename_axis("model").to_csv(OUTPUT_DIR / "evaluation_metrics.csv")
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
    order_col="importance_mean" if "importance_mean" in feature_importance.columns else "importance"
    feature_importance.sort_values(order_col, ascending=False).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
