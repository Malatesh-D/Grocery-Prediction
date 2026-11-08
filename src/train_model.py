"""Leakage-aware benchmark, selection, evaluation, and artifact persistence."""
from datetime import datetime, timezone
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.evaluate_model import regression_metrics, save_model_plots, save_eda
from src.preprocess import build_preprocessor
from src.utils import MODEL_DIR, OUTPUT_DIR, ensure_directories, dataset_fingerprint
from src.validate_data import validate_training_data

SEED = 42
TARGET = "monthly_grocery_expenditure_inr"

def train_frame(df, provenance="Synthetic demonstration data", data_kind="synthetic", source_name="grocery_data.csv"):
    threadpool_limits(limits=1)
    ensure_directories()
    frame = df.copy()
    report = validate_training_data(frame, TARGET)
    features = report["features"]
    if data_kind == "synthetic":
        save_eda(frame, features, TARGET)
    X, y = frame[features], frame[TARGET]
    split_group=frame["_split_group"] if "_split_group" in frame else None
    split_time=pd.to_datetime(frame["_split_time"],errors="coerce") if "_split_time" in frame else None
    if split_group is not None:
        splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED)
        train_idx,test_idx=next(splitter.split(X,y,split_group))
        folds=min(5 if len(train_idx)>=100 else 3,len(pd.unique(split_group.iloc[train_idx])))
        if folds<2: raise ValueError("Group-aware validation needs at least two distinct household groups in training data.")
        cv=GroupKFold(n_splits=folds); groups=split_group.iloc[train_idx]
        split_strategy="Group-aware holdout and GroupKFold by household identifier"
    elif split_time is not None:
        if split_time.isna().any(): raise ValueError("Time column contains invalid or missing dates.")
        order=np.argsort(split_time.to_numpy()); cut=int(len(order)*.8)
        train_idx,test_idx=order[:cut],order[cut:]
        folds=5 if len(train_idx)>=100 else 3
        cv=TimeSeriesSplit(n_splits=folds); groups=None
        split_strategy="Chronological last-20% holdout and expanding-window TimeSeriesSplit"
    else:
        train_idx,test_idx=train_test_split(np.arange(len(frame)),test_size=.2,random_state=SEED)
        folds=5 if len(train_idx)>=100 else 3
        cv=KFold(n_splits=folds,shuffle=True,random_state=SEED); groups=None
        split_strategy="Random independent-row 80/20 holdout; shuffled KFold on training split"
    X_train,X_test=X.iloc[train_idx],X.iloc[test_idx]
    y_train,y_test=y.iloc[train_idx],y.iloc[test_idx]
    candidates = {
        "Mean baseline": Pipeline([("preprocessor", build_preprocessor(features)), ("model", DummyRegressor(strategy="mean"))]),
        "Linear Regression": Pipeline([("preprocessor", build_preprocessor(features)), ("model", LinearRegression())]),
        "Random Forest": Pipeline([("preprocessor", build_preprocessor(features)), ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=.85, random_state=SEED, n_jobs=1))]),
        "HistGradientBoosting": Pipeline([("preprocessor", build_preprocessor(features)), ("model", HistGradientBoostingRegressor(max_iter=150, l2_regularization=1.0, random_state=SEED))]),
    }
    cv_scores, cv_mae, fitted, test_metrics, test_predictions = {}, {}, {}, {}, {}
    for name, pipeline in candidates.items():
        rmse_scores = -cross_val_score(pipeline, X_train, y_train, cv=cv, groups=groups, scoring="neg_root_mean_squared_error", n_jobs=1)
        mae_scores = -cross_val_score(pipeline, X_train, y_train, cv=cv, groups=groups, scoring="neg_mean_absolute_error", n_jobs=1)
        cv_scores[name] = {"RMSE_mean": float(rmse_scores.mean()), "RMSE_std": float(rmse_scores.std()),
                          "MAE_mean": float(mae_scores.mean()), "folds": folds}
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    # Selection only sees training-fold CV. The holdout remains isolated until final report.
    selected = min(candidates, key=lambda n: (cv_scores[n]["RMSE_mean"], cv_scores[n]["MAE_mean"]))
    for name, pipeline in fitted.items():
        predictions = pipeline.predict(X_test)
        test_metrics[name] = regression_metrics(y_test, predictions)
        test_predictions[name] = predictions
    chosen = fitted[selected]
    joblib.dump(chosen, MODEL_DIR / "grocery_spend_pipeline.joblib")
    digest = hashlib.sha256((MODEL_DIR / "grocery_spend_pipeline.joblib").read_bytes()).hexdigest()
    prediction_path=OUTPUT_DIR / "test_predictions.csv"
    if data_kind == "synthetic":
        pd.DataFrame({"actual_inr": y_test.to_numpy(), "predicted_inr": test_predictions[selected],
                      "residual_inr": y_test.to_numpy() - test_predictions[selected]}).to_csv(prediction_path, index=False)
    elif prediction_path.exists():
        prediction_path.unlink()
    importance = permutation_importance(chosen, X_test, y_test, n_repeats=10, random_state=SEED,
                                        scoring="neg_mean_absolute_error", n_jobs=1)
    importance_df = pd.DataFrame({"feature": features, "importance_mean": importance.importances_mean,
                                  "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    save_model_plots(y_test.to_numpy(), test_predictions[selected], test_metrics, importance_df)
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_metrics": test_metrics, "cross_validation": cv_scores}, f, indent=2)
    detail = {"selected_model": selected, "selection_rule": "Lowest mean training-only CV RMSE; CV MAE tie-breaker",
              "split_strategy": split_strategy + f"; {folds}-fold CV on training split",
              "random_seed": SEED, "test_size": .2, "training_records": len(X_train), "test_records": len(X_test),
              "features": features, "target": TARGET, "data_kind": data_kind, "provenance": provenance,
              "source_name": source_name, "dataset_fingerprint_sha256": dataset_fingerprint(frame),
              "trained_at_utc": datetime.now(timezone.utc).isoformat(), "artifact_sha256": digest,
              "artifact_version": "1.0", "feature_ranges": {c:{"min":float(X[c].min()),"max":float(X[c].max())} for c in features if X[c].notna().any()},
              "feature_defaults": {c:float(X_train[c].median()) for c in features if X_train[c].notna().any()},
              "quality_report": report}
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    (MODEL_DIR / "feature_schema.json").write_text(json.dumps({"features":features,"target":TARGET,"target_unit":"INR per month",
        "feature_ranges":detail["feature_ranges"],"feature_defaults":detail["feature_defaults"],"data_kind":data_kind},indent=2),encoding="utf-8")
    (OUTPUT_DIR / "training_config.json").write_text(json.dumps({"seed":SEED,"test_fraction":.2,"cv_folds":folds,
        "selection_metric":"mean CV RMSE with CV MAE tie-break","models":list(candidates),
        "model_parameters":{"Mean baseline":{"strategy":"mean"},"Random Forest":{"n_estimators":250,"min_samples_leaf":3,"max_features":.85,"random_state":SEED},
        "HistGradientBoosting":{"max_iter":150,"l2_regularization":1.0,"random_state":SEED}},"feature_schema":features}, indent=2), encoding="utf-8")
    summary = {"test_metrics":test_metrics,"cross_validation":cv_scores,"details":detail}
    print(json.dumps(summary, indent=2))
    return summary

def train():
    from src.utils import DATA_PATH, DATA_DIR
    manifest_path=DATA_DIR/"dataset_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Dataset provenance manifest is missing. Use the authorized upload workflow or generate a labeled synthetic dataset first.")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    df = pd.read_csv(DATA_PATH)
    return train_frame(df, manifest["provenance"], manifest["data_kind"], manifest["source_name"])

if __name__ == "__main__":
    train()
