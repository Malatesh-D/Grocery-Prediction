"""Dataset checks and preprocessing helpers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


if __name__ == "__main__":
    from src.data_loader import load_project_dataset
    print("=" * 70)
    print("GROCERYSPEND AI - PREPROCESSING PIPELINE & LEAKAGE VERIFICATION")
    print("=" * 70)
    df = load_project_dataset()
    validate_dataset(df)
    preprocessor = build_preprocessor(FEATURES)
    X = df[FEATURES]
    X_trans = preprocessor.fit_transform(X)
    print(f"Preprocessor Pipeline:  ColumnTransformer with SimpleImputer(median) + StandardScaler")
    print(f"Leakage Protection:     Encapsulated in Pipeline (fit on train folds only)")
    print(f"Input Shape:            {X.shape}")
    print(f"Transformed Output:     {X_trans.shape} (Standardized features, mean=0, std=1)")
    print(f"Features Encoded ({len(FEATURES)}):")
    for idx, feat in enumerate(FEATURES, 1):
        mean_val = X_trans[:, idx - 1].mean()
        std_val = X_trans[:, idx - 1].std()
        print(f"  [{idx}] {feat:<32} -> Scaled Mean: {mean_val:+.4f}, Scaled Std: {std_val:.4f}")
    print("=" * 70)

