"""Schema mapping and data quality checks for household expenditure CSVs."""
import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "household_size": "Household size (people)",
    "monthly_household_income_inr": "Monthly household income (INR)",
    "shopping_trips_per_month": "Grocery trips per month",
    "avg_items_per_trip": "Average items per trip",
    "branded_spending_pct": "Branded spending (%)",
    "fresh_produce_spending_pct": "Fresh produce spending (%)",
    "online_shopping_pct": "Online shopping (%)",
    "previous_month_expenditure_inr": "Previous month expenditure (INR)",
}
PERCENT_FEATURES = {"branded_spending_pct", "fresh_produce_spending_pct", "online_shopping_pct"}

def map_columns(df, mapping, target_source):
    """Rename chosen source columns to canonical input names; omit unavailable features."""
    if target_source not in df.columns:
        raise ValueError("Choose a target column that exists in the uploaded CSV.")
    used = [v for v in mapping.values() if v]
    if len(used) != len(set(used)) or target_source in used:
        raise ValueError("Each source column can map to only one feature and the target.")
    reverse = {source: canonical for canonical, source in mapping.items() if source}
    out = df.rename(columns=reverse).copy()
    out = out.rename(columns={target_source: "monthly_grocery_expenditure_inr"})
    keep = list(reverse.values()) + ["monthly_grocery_expenditure_inr"]
    out = out[keep]
    return out

def validate_training_data(df, target="monthly_grocery_expenditure_inr", min_rows=30):
    if target not in df:
        raise ValueError(f"Target column '{target}' is missing.")
    features = [c for c in df.columns if c != target and not c.startswith("_split_")]
    if not features:
        raise ValueError("Map at least one input feature.")
    all_missing=[col for col in features if df[col].isna().all()]
    if all_missing:
        raise ValueError(f"Selected feature(s) have no observed values: {all_missing}")
    if len(df) < min_rows:
        raise ValueError(f"At least {min_rows} rows are required; found {len(df)}.")
    if df[target].isna().any():
        raise ValueError("Target values cannot be missing; remove or correct those rows.")
    for col in [target] + features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[target].isna().any() or not np.isfinite(df[target]).all():
        raise ValueError("Target must contain finite numeric expenditure values in INR.")
    if (df[target] < 0).any():
        raise ValueError("Negative target expenditure is invalid.")
    if df.duplicated().any():
        raise ValueError("Duplicate records were found. Review/deduplicate the source before training.")
    for col in features:
        observed = df[col].dropna()
        if not observed.empty and not np.isfinite(observed).all():
            raise ValueError(f"Feature '{col}' contains non-finite values.")
        if col in PERCENT_FEATURES and not observed.between(0, 100).all():
            raise ValueError(f"'{col}' must be in the range 0–100 percent.")
        if col == "household_size" and ((observed < 1).any() or (observed % 1 != 0).any()):
            raise ValueError("Household size must be positive whole people.")
        if col.endswith("_inr") and (observed < 0).any():
            raise ValueError(f"'{col}' cannot contain negative INR values.")
    if df[features].isna().all(axis=None):
        raise ValueError("No observed values are available in the selected features.")
    return {"rows": len(df), "features": features, "missing_by_column": df.isna().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()), "target_min_inr": float(df[target].min()),
            "target_max_inr": float(df[target].max())}
