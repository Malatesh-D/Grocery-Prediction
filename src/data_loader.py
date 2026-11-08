"""CSV ingestion helpers; uploaded contents are processed in memory."""
from pathlib import Path
import pandas as pd

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

def load_project_dataset(path):
    return load_csv(Path(path))
