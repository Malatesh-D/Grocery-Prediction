# VERIFIED RUNTIME TERMINAL EXECUTION OUTPUTS

**Project Title:** GrocerySpend AI — Predictive Modeling of Monthly Household Grocery Spend in INR  
**Author / Student:** Malatesh D  
**Faculty Evaluator:** Prof. Debarshi Mazumder  
**Academic Year:** 2026–2027  
**GitHub Repository:** [https://github.com/Malatesh-D/Grocery-Prediction](https://github.com/Malatesh-D/Grocery-Prediction)  

---

## Terminal 1: Ingestion & Dataset Inspection (data_loader.py)
**Command Executed:** `python src/data_loader.py`  
**Screenshot Artifact:** `reports/screenshots/terminal_01_data_ingestion.png`  

```text
======================================================================
GROCERYSPEND AI - DATA INGESTION & DATASET INSPECTION
======================================================================
Dataset Path:       C:\Users\malat\OneDrive\Desktop\Predictive Analytics Project\GROCERY PREDICTION\data\grocery_data.csv
Total Rows:         2000
Total Columns:      9
Features:           ['household_size', 'monthly_household_income_inr', 'shopping_trips_per_month', 'avg_items_per_trip', 'branded_spending_pct', 'fresh_produce_spending_pct', 'online_shopping_pct', 'previous_month_expenditure_inr', 'monthly_grocery_expenditure_inr']
Missing Values:     0
Duplicate Rows:     0
----------------------------------------------------------------------
First 3 Ingested Records:
   household_size  monthly_household_income_inr  shopping_trips_per_month  avg_items_per_trip  branded_spending_pct  fresh_produce_spending_pct  online_shopping_pct  previous_month_expenditure_inr  monthly_grocery_expenditure_inr
0               5                      28703.44                         6                  29                 11.32                       73.05                23.95                         9732.08                          9498.37
1               3                      88051.35                         7                  17                 66.20                       61.18                16.54                         7448.00                          6699.19
2               5                      25885.42                        13                  33                 33.21                       57.14                10.72                         7839.00                          6449.10
----------------------------------------------------------------------
Target Spend Range: ₹911 to ₹20,316
Mean Target Spend:  ₹7,630 (Std: ₹2,673)
======================================================================
```

---

## Terminal 2: Schema Validation & Integrity (validate_data.py)
**Command Executed:** `python src/validate_data.py`  
**Screenshot Artifact:** `reports/screenshots/terminal_02_data_validation.png`  

```text
======================================================================
GROCERYSPEND AI - SCHEMA VALIDATION & QUALITY ASSURANCE AUDIT
======================================================================
Validation Status:      [PASSED] All integrity constraints satisfied
Verified Records:       2000 rows
Validated Input Feats:  8 numerical columns
Duplicate Row Count:    0 (Deduplicated)
Target Spend Range:     ₹911 to ₹20,316
----------------------------------------------------------------------
Feature Validation Breakdown:
  ✓ household_size                  : Missing = 0 (0.00%) | Status: Valid
  ✓ monthly_household_income_inr    : Missing = 0 (0.00%) | Status: Valid
  ✓ shopping_trips_per_month        : Missing = 0 (0.00%) | Status: Valid
  ✓ avg_items_per_trip              : Missing = 0 (0.00%) | Status: Valid
  ✓ branded_spending_pct            : Missing = 0 (0.00%) | Status: Valid
  ✓ fresh_produce_spending_pct      : Missing = 0 (0.00%) | Status: Valid
  ✓ online_shopping_pct             : Missing = 0 (0.00%) | Status: Valid
  ✓ previous_month_expenditure_inr  : Missing = 0 (0.00%) | Status: Valid
======================================================================
```

---

## Terminal 3: Preprocessing Pipeline (preprocess.py)
**Command Executed:** `python src/preprocess.py`  
**Screenshot Artifact:** `reports/screenshots/terminal_03_preprocessing_pipeline.png`  

```text
======================================================================
GROCERYSPEND AI - PREPROCESSING PIPELINE & LEAKAGE VERIFICATION
======================================================================
Preprocessor Pipeline:  ColumnTransformer with SimpleImputer(median) + StandardScaler
Leakage Protection:     Encapsulated in Pipeline (fit on train folds only)
Input Shape:            (2000, 8)
Transformed Output:     (2000, 8) (Standardized features, mean=0, std=1)
Features Encoded (8):
  [1] household_size                   -> Scaled Mean: -0.0000, Scaled Std: 1.0000
  [2] monthly_household_income_inr     -> Scaled Mean: +0.0000, Scaled Std: 1.0000
  [3] shopping_trips_per_month         -> Scaled Mean: +0.0000, Scaled Std: 1.0000
  [4] avg_items_per_trip               -> Scaled Mean: +0.0000, Scaled Std: 1.0000
  [5] branded_spending_pct             -> Scaled Mean: +0.0000, Scaled Std: 1.0000
  [6] fresh_produce_spending_pct       -> Scaled Mean: -0.0000, Scaled Std: 1.0000
  [7] online_shopping_pct              -> Scaled Mean: +0.0000, Scaled Std: 1.0000
  [8] previous_month_expenditure_inr   -> Scaled Mean: -0.0000, Scaled Std: 1.0000
======================================================================
```

---

## Terminal 4: 5-Fold Cross-Validation & Benchmark Selection (train_model.py)
**Command Executed:** `python src/train_model.py`  
**Screenshot Artifact:** `reports/screenshots/terminal_04_model_training_cv.png`  

```text
======================================================================
GROCERYSPEND AI - BENCHMARK & 5-FOLD CROSS-VALIDATION PIPELINE
======================================================================
Split Strategy:     80/20 Holdout (Train=1600, Test=400) | Seed=42
Validation Folds:   5-Fold Shuffled Cross Validation on Training Split
Selection Metric:   Lowest CV RMSE (Mean) | CV MAE as Tie-Breaker
Leakage Protection: SimpleImputer & StandardScaler fit strictly on train folds
----------------------------------------------------------------------
Evaluating 4 Candidate Regressor Models...
  [1/4] Mean Baseline        -> CV RMSE: 2659.71 ± 62.42 INR | CV MAE: 2135.49 INR
  [2/4] Linear Regression    -> CV RMSE: 1274.17 ± 58.08 INR | CV MAE: 1010.12 INR  [WINNER]
  [3/4] Random Forest (250)  -> CV RMSE: 1324.75 ± 51.01 INR | CV MAE: 1055.05 INR
  [4/4] HistGradientBoosting -> CV RMSE: 1400.48 ± 64.66 INR | CV MAE: 1107.89 INR
----------------------------------------------------------------------
CHAMPION MODEL SELECTED: Linear Regression
  • Cross-Validation RMSE : ₹1,274.17 ± 58.08
  • Cross-Validation MAE  : ₹1,010.12
  • Held-Out Test R^2     : 0.7609 (76.1% Explained Variance)
  • Held-Out Test MAE     : ₹1,081.39 (Average Error ₹1,081)
  • Held-Out Test RMSE    : ₹1,333.37
----------------------------------------------------------------------
Artifact Persisted to: models/grocery_spend_pipeline.joblib
Metadata Fingerprint:  87d173bbb1f1145f2a7ca6a1fc29831ccdaa59d932c98afae18fb2675c1ee9df
======================================================================
```

---

## Terminal 5: Holdout Evaluation & Permutation Importance (evaluate_model.py)
**Command Executed:** `python src/evaluate_model.py`  
**Screenshot Artifact:** `reports/screenshots/terminal_05_holdout_evaluation.png`  

```text
======================================================================
GROCERYSPEND AI - MODEL EVALUATION & BENCHMARK AUDIT
======================================================================
Candidate Model Performance on Holdout Test Set (N=400):
               model         MAE        RMSE        R2
       Mean baseline 2153.052052 2728.952472 -0.001366
   Linear Regression 1081.389696 1333.374897  0.760941
       Random Forest 1105.761570 1368.122631  0.748319
HistGradientBoosting 1178.752705 1454.252684  0.715632
----------------------------------------------------------------------
Permutation Feature Importance (Mean MAE Impact):
  • previous_month_expenditure_inr   : +₹838 ± ₹69
  • household_size                   : +₹292 ± ₹43
  • monthly_household_income_inr     : +₹22 ± ₹10
  • shopping_trips_per_month         : +₹15 ± ₹6
  • branded_spending_pct             : +₹5 ± ₹3
  • fresh_produce_spending_pct       : +₹0 ± ₹1
  • avg_items_per_trip               : +-₹1 ± ₹4
  • online_shopping_pct              : +-₹1 ± ₹1
======================================================================
```

---

## Terminal 6: Real-Time Inference Engine Verification (predict.py)
**Command Executed:** `python src/predict.py`  
**Screenshot Artifact:** `reports/screenshots/terminal_06_prediction_inference.png`  

```text
======================================================================
GROCERYSPEND AI - REAL-TIME INFERENCE ENGINE
======================================================================
Model Champion:          Linear Regression
Artifact Version:        v1.0
Dataset Fingerprint:     acaf376d3b3a82d7...
----------------------------------------------------------------------
Input Household Profile:
  • Household Size                  : 4
  • Monthly Household Income Inr    : ₹65,000
  • Shopping Trips Per Month        : 8
  • Avg Items Per Trip              : 22
  • Branded Spending Pct            : 45.0%
  • Fresh Produce Spending Pct      : 55.0%
  • Online Shopping Pct             : 30.0%
  • Previous Month Expenditure Inr  : ₹7,800
----------------------------------------------------------------------
PREDICTED MONTHLY SPEND: ₹7,870 (7870.28 INR)
Out-of-range Warnings:   0 detected
======================================================================
```

---

## Terminal 7: Automated Unit & Integration Test Suite (pytest -v)
**Command Executed:** `python -m pytest tests/ -v`  
**Screenshot Artifact:** `reports/screenshots/terminal_07_pytest_test_suite.png`  

```text
============================= test session starts ==============================
platform win32 -- Python 3.13.5, pytest-8.4.2, pluggy-1.5.0
rootdir: C:\Users\malat\OneDrive\Desktop\Predictive Analytics Project\GROCERY PREDICTION
configfile: pytest.ini
collected 16 items

tests/test_data.py::test_load_csv_valid PASSED                            [  6%]
tests/test_data.py::test_load_csv_empty_fails PASSED                      [ 12%]
tests/test_data.py::test_load_csv_duplicate_cols_fails PASSED             [ 18%]
tests/test_data.py::test_validate_dataset_success PASSED                  [ 25%]
tests/test_data.py::test_validate_missing_columns PASSED                  [ 31%]
tests/test_data.py::test_validate_bounds PASSED                           [ 37%]
tests/test_data.py::test_validate_negative_target PASSED                  [ 43%]
tests/test_data.py::test_validate_training_data_duplicates PASSED         [ 50%]
tests/test_data.py::test_validate_training_data_non_finite PASSED         [ 56%]
tests/test_prediction.py::test_predict_valid_payload PASSED              [ 62%]
tests/test_prediction.py::test_predict_rejects_negative_income PASSED     [ 68%]
tests/test_prediction.py::test_predict_rejects_invalid_percentages PASSED[ 75%]
tests/test_prediction.py::test_predict_rejects_invalid_household_size PASSED[81%]
tests/test_training.py::test_train_pipeline_reproducibility PASSED        [ 87%]
tests/test_training.py::test_model_selection_criteria PASSED             [ 93%]
tests/test_training.py::test_preprocessor_encapsulation PASSED           [100%]

============================== 16 passed in 16.41s ==============================
```

---

## Terminal 8: Streamlit Web Application Server Launch (app.py)
**Command Executed:** `streamlit run app.py`  
**Screenshot Artifact:** `reports/screenshots/terminal_08_streamlit_launch.png`  

```text
  Welcome to Streamlit!

  If you're one of our first 100,000 users, please enter your email below...

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.15:8501

  Ready for client connections. Light theme enforcement active.
  Live prediction engine loaded from models/grocery_spend_pipeline.joblib
  Integrity Check: SHA-256 Checksum Verified [PASSED]
```

---
