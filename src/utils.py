"""Shared paths, feature definitions, and formatting helpers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = DATA_DIR / "grocery_data.csv"
MODEL_PATH = MODEL_DIR / "grocery_spend_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
TARGET = "monthly_grocery_expenditure_inr"
FEATURES = [
    "household_size", "monthly_household_income_inr", "shopping_trips_per_month",
    "avg_items_per_trip", "branded_spending_pct", "fresh_produce_spending_pct",
    "online_shopping_pct", "previous_month_expenditure_inr",
]
BOUNDS = {
    "household_size": (1, 12), "monthly_household_income_inr": (5000, 500000),
    "shopping_trips_per_month": (1, 31), "avg_items_per_trip": (1, 100),
    "branded_spending_pct": (0, 100), "fresh_produce_spending_pct": (0, 100),
    "online_shopping_pct": (0, 100), "previous_month_expenditure_inr": (500, 150000),
}

def ensure_directories():
    for path in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)

def dataset_fingerprint(df):
    import hashlib
    import pandas as pd
    fingerprint_frame = df[[c for c in df.columns if not c.startswith("_split_")]]
    payload = pd.util.hash_pandas_object(fingerprint_frame, index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()

def inr(value):
    """Format a number using Indian digit grouping."""
    value = round(float(value))
    sign = "-" if value < 0 else ""
    digits = str(abs(value))
    if len(digits) <= 3:
        return f"{sign}₹{digits}"
    last = digits[-3:]
    head = digits[:-3]
    groups = []
    while head:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return f"{sign}₹{','.join(groups)},{last}"
