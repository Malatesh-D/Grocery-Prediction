"""Safe-by-integrity-check inference against the project's trusted model artifact."""
import hashlib
import json
import math
import joblib
import pandas as pd
from src.utils import MODEL_PATH, METADATA_PATH

class ModelArtifactError(RuntimeError):
    """The model or its metadata is missing, corrupt, or incompatible."""

def load_artifact(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    if not model_path.exists() or not metadata_path.exists():
        raise ModelArtifactError("Trained model or metadata is missing. Train a dataset first.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != metadata.get("artifact_sha256"):
            raise ModelArtifactError("Model artifact checksum does not match its metadata; retrain from a trusted dataset.")
        if metadata.get("artifact_version") != "1.0" or not metadata.get("features"):
            raise ModelArtifactError("Model metadata version or feature schema is unsupported.")
        model = joblib.load(model_path)
        return model, metadata
    except ModelArtifactError:
        raise
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model artifact: {exc}") from exc

def validate_inputs(values, metadata=None):
    features = metadata["features"] if metadata else list(values)
    missing = [name for name in features if name not in values]
    if missing:
        raise ValueError(f"Missing inputs: {', '.join(missing)}")
    clean = {}
    for name in features:
        try: value = float(values[name])
        except (TypeError, ValueError): raise ValueError(f"{name} must be a number")
        if not math.isfinite(value): raise ValueError(f"{name} must be a finite number")
        if name in {"branded_spending_pct", "fresh_produce_spending_pct", "online_shopping_pct"} and not 0 <= value <= 100:
            raise ValueError(f"{name} must be between 0 and 100 percent")
        if name == "household_size" and (value < 1 or not value.is_integer()):
            raise ValueError("household_size must be a positive whole number")
        if (name.endswith("_inr") or name == "monthly_household_income_inr") and value < 0:
            raise ValueError(f"{name} cannot be negative")
        clean[name] = value
    return clean

def predict_expenditure(values, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model, metadata = load_artifact(model_path, metadata_path)
    clean = validate_inputs(values, metadata)
    pred = float(model.predict(pd.DataFrame([clean], columns=metadata["features"]))[0])
    ranges = metadata.get("feature_ranges", {})
    out_of_range = [name for name,value in clean.items() if name in ranges and not ranges[name]["min"] <= value <= ranges[name]["max"]]
    return {"prediction_inr": pred, "out_of_training_range": out_of_range,
            "model_version": metadata["artifact_version"],
            "dataset_fingerprint": metadata["dataset_fingerprint_sha256"],
            "data_kind": metadata["data_kind"], "model_name": metadata["selected_model"]}
