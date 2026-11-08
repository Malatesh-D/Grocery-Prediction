"""Safe-by-integrity-check inference against the project's trusted model artifact."""
import hashlib
import json
import math
import joblib
import numpy as np
import pandas as pd
from src.utils import MODEL_PATH, METADATA_PATH

class ModelArtifactError(RuntimeError):
    """The model or its metadata is missing, corrupt, or incompatible."""

def _patch_imputers(estimator):
    """Ensure cross-version compatibility for unpickled SimpleImputer (e.g. _fill_dtype added in scikit-learn 1.5+)."""
    try:
        if hasattr(estimator, "named_steps"):
            for step in estimator.named_steps.values():
                _patch_imputers(step)
        if hasattr(estimator, "transformers_"):
            for item in estimator.transformers_:
                if len(item) >= 2:
                    _patch_imputers(item[1])
        if hasattr(estimator, "named_transformers_"):
            for trans in estimator.named_transformers_.values():
                _patch_imputers(trans)
        if hasattr(estimator, "statistics_") and not hasattr(estimator, "_fill_dtype"):
            estimator._fill_dtype = getattr(estimator.statistics_, "dtype", np.float64)
    except Exception:
        pass

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
        _patch_imputers(model)
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
    input_df = pd.DataFrame([clean], columns=metadata["features"])
    try:
        pred = float(model.predict(input_df)[0])
    except AttributeError as err:
        _patch_imputers(model)
        try:
            pred = float(model.predict(input_df)[0])
        except Exception:
            from src.train_model import train
            train()
            model, metadata = load_artifact(model_path, metadata_path)
            _patch_imputers(model)
            pred = float(model.predict(input_df)[0])
    ranges = metadata.get("feature_ranges", {})
    out_of_range = [name for name,value in clean.items() if name in ranges and not ranges[name]["min"] <= value <= ranges[name]["max"]]
    return {"prediction_inr": pred, "out_of_training_range": out_of_range,
            "model_version": metadata["artifact_version"],
            "dataset_fingerprint": metadata["dataset_fingerprint_sha256"],
            "data_kind": metadata["data_kind"], "model_name": metadata["selected_model"]}
