import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from src.generate_data import generate_dataset
import src.train_model as training
import src.evaluate_model as evaluation
import src.utils as utils
train_frame, SEED = training.train_frame, training.SEED

def isolate_artifacts(monkeypatch,name):
    folder=Path(__file__).parent/f".{name}_artifacts"
    shutil.rmtree(folder,ignore_errors=True); (folder/"models").mkdir(parents=True); (folder/"outputs").mkdir()
    monkeypatch.setattr(training,"MODEL_DIR",folder/"models"); monkeypatch.setattr(training,"OUTPUT_DIR",folder/"outputs")
    monkeypatch.setattr(evaluation,"OUTPUT_DIR",folder/"outputs")
    monkeypatch.setattr(utils,"MODEL_DIR",folder/"models"); monkeypatch.setattr(utils,"OUTPUT_DIR",folder/"outputs")
    return folder

def test_reproducible_split():
    x=np.arange(100)
    a,b=train_test_split(x,test_size=.2,random_state=SEED)
    c,d=train_test_split(x,test_size=.2,random_state=SEED)
    assert np.array_equal(a,c) and np.array_equal(b,d)

def test_training_produces_comparison_and_test_metrics(monkeypatch):
    folder=isolate_artifacts(monkeypatch,"training")
    try:
        result=train_frame(generate_dataset(80),"Test-only synthetic fixture","synthetic","test_fixture.csv")
        assert result["details"]["test_records"]==16
        assert "Mean baseline" in result["test_metrics"]
        assert "Linear Regression" in result["cross_validation"]
        assert result["details"]["dataset_fingerprint_sha256"]
        assert (folder/"models"/"feature_schema.json").exists()
    finally: shutil.rmtree(folder,ignore_errors=True)

def test_training_supports_group_aware_splitting(monkeypatch):
    folder=isolate_artifacts(monkeypatch,"group_training")
    try:
        df=generate_dataset(80)
        df["_split_group"]=[f"household_{i//2}" for i in range(len(df))]
        result=train_frame(df,"Group split fixture","synthetic","group_fixture.csv")
        assert "GroupKFold" in result["details"]["split_strategy"]
        assert "_split_group" not in result["details"]["features"]
    finally: shutil.rmtree(folder,ignore_errors=True)
