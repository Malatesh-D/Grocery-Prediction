import pandas as pd
import pytest
from src.generate_data import generate_dataset
from src.preprocess import validate_dataset
from src.validate_data import validate_training_data, map_columns
from src.utils import FEATURES, TARGET

def test_generated_data_valid():
    df = generate_dataset(1500, 42)
    assert df.shape == (1500, 9)
    assert validate_dataset(df)
    assert (df[FEATURES + [TARGET]] >= 0).all().all()
    for col in ["branded_spending_pct", "fresh_produce_spending_pct", "online_shopping_pct"]:
        assert df[col].between(0, 100).all()

def test_generation_reproducible():
    pd.testing.assert_frame_equal(generate_dataset(50, 3), generate_dataset(50, 3))

def test_validation_rejects_bad_percentage():
    df = generate_dataset(20)
    df.loc[0, "online_shopping_pct"] = 101
    with pytest.raises(ValueError): validate_dataset(df)

def test_mapping_adapts_to_available_source_columns():
    source = pd.DataFrame({"family_count":[2,3],"food_spend":[4000,5000]})
    mapped=map_columns(source,{"household_size":"family_count"},"food_spend")
    assert list(mapped.columns)==["household_size",TARGET]

def test_training_validation_allows_missing_features_for_pipeline_imputation():
    df=generate_dataset(40)
    df.loc[:4,"monthly_household_income_inr"]=None
    assert validate_training_data(df)["missing_by_column"]["monthly_household_income_inr"]==5

def test_training_validation_rejects_negative_target():
    df=generate_dataset(40); df.loc[0,TARGET]=-1
    with pytest.raises(ValueError,match="Negative"): validate_training_data(df)

def test_training_validation_rejects_bad_percentage():
    df=generate_dataset(40); df.loc[0,"online_shopping_pct"]=101
    with pytest.raises(ValueError,match="online_shopping_pct"): validate_training_data(df)

def test_training_validation_requires_target():
    with pytest.raises(ValueError,match="monthly_grocery_expenditure_inr"):
        validate_training_data(pd.DataFrame({"some_input":[1]*40}))

def test_training_validation_rejects_bad_target_values():
    df=generate_dataset(40).astype({TARGET:"object"}); df.loc[0,TARGET]="unknown"
    with pytest.raises(ValueError,match="Target"): validate_training_data(df)
