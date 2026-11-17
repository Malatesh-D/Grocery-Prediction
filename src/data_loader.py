"""CSV ingestion helpers; uploaded contents are processed in memory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils import DATA_PATH, inr

def load_csv(source):
    if hasattr(source, "seek"):
        source.seek(0)
    frame = pd.read_csv(source)
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    return frame

def load_project_dataset(path=DATA_PATH):
    return load_csv(Path(path))

if __name__ == "__main__":
    print("=" * 70)
    print("GROCERYSPEND AI - DATA INGESTION & DATASET INSPECTION")
    print("=" * 70)
    df = load_project_dataset()
    print(f"Dataset Path:       {DATA_PATH}")
    print(f"Total Rows:         {len(df)}")
    print(f"Total Columns:      {len(df.columns)}")
    print(f"Features:           {list(df.columns)}")
    print(f"Missing Values:     {df.isna().sum().sum()}")
    print(f"Duplicate Rows:     {df.duplicated().sum()}")
    print("-" * 70)
    print("First 3 Ingested Records:")
    print(df.head(3).to_string())
    print("-" * 70)
    print(f"Target Spend Range: {inr(df['monthly_grocery_expenditure_inr'].min())} to {inr(df['monthly_grocery_expenditure_inr'].max())}")
    print(f"Mean Target Spend:  {inr(df['monthly_grocery_expenditure_inr'].mean())} (Std: {inr(df['monthly_grocery_expenditure_inr'].std())})")
    print("=" * 70)

