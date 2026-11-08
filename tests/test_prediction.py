import json
import shutil
import hashlib
import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
import pytest
from src.predict import predict_expenditure, load_artifact, ModelArtifactError, validate_inputs
from src.utils import MODEL_DIR

def test_prediction_matches_feature_order_and_returns_provenance():
    temp=MODEL_DIR/".test_artifact_tmp"; temp.mkdir(exist_ok=True)
    model_path=temp/"pipeline.joblib"; meta_path=temp/"metadata.json"
    features=["household_size","previous_month_expenditure_inr"]
    model=Pipeline([("model",DummyRegressor(strategy="constant",constant=1234.))])
    model.fit(pd.DataFrame([[2,4000]],columns=features),[1234.])
    try:
        joblib.dump(model,model_path)
        metadata={"artifact_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),"artifact_version":"1.0","features":features,"feature_ranges":{"household_size":{"min":1,"max":4}},"selected_model":"test","dataset_fingerprint_sha256":"abc","data_kind":"synthetic"}
        meta_path.write_text(json.dumps(metadata))
        result=predict_expenditure({"household_size":6,"previous_month_expenditure_inr":4000},model_path,meta_path)
        assert result["model_name"]=="test" and result["dataset_fingerprint"]=="abc"
        assert result["prediction_inr"]==1234. and result["out_of_training_range"]==["household_size"]
    finally: shutil.rmtree(temp,ignore_errors=True)

def test_input_validation_rejects_bad_percentages():
    with pytest.raises(ValueError): validate_inputs({"branded_spending_pct": 101})

def test_input_validation_rejects_invalid_household_size():
    with pytest.raises(ValueError): validate_inputs({"household_size": 1.5})

def test_missing_and_corrupted_artifacts_are_reported():
    temp=MODEL_DIR/".test_artifact_tmp"; temp.mkdir(exist_ok=True)
    model=temp/"bad.joblib"; meta=temp/"meta.json"
    try:
        model.write_bytes(b"broken"); meta.write_text(json.dumps({"artifact_sha256":"wrong","artifact_version":"1.0","features":["x"]}))
        with pytest.raises(ModelArtifactError): load_artifact(model,meta)
        with pytest.raises(ModelArtifactError): load_artifact(temp/"missing.joblib",meta)
    finally: shutil.rmtree(temp,ignore_errors=True)
