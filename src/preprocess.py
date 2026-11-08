"""Dataset checks and preprocessing helpers."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils import FEATURES, TARGET, BOUNDS

def validate_dataset(df):
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Required columns contain missing values")
    if df.duplicated().any():
        raise ValueError("Dataset contains duplicate rows")
    for col, (low, high) in BOUNDS.items():
        if not df[col].between(low, high).all():
            raise ValueError(f"{col} must be between {low} and {high}")
    if (df[TARGET] <= 0).any():
        raise ValueError("Target expenditure must be positive")
    return True

def build_preprocessor(features=None):
    features = FEATURES if features is None else list(features)
    return ColumnTransformer([("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), features)], remainder="drop")
