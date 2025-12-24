# src/train_baseline.py
"""
Simple baseline training script (improved):
- Loads features/labels from prefix
- Trains RandomForest, SVM (RBF), LogisticRegression
- Saves best pipeline (scaler + model)
- Saves classification report, confusion matrix, predictions CSV
- Optionally saves RF feature importances (if available)


"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def load_data(prefix):
    X = np.load(prefix + "_features.npy")
    y = np.load(prefix + "_labels.npy")
    # try to load csv header for feature names (optional)
    feat_names = None
    csv_path = prefix + "_features.csv"
    if os.path.exists(csv_path):
        try:
            dfh = pd.read_csv(csv_path, nrows=0)
            feat_names = list(dfh.columns[3:])  # filepath,label,scanner_name first cols
        except Exception:
            feat_names = None
    return X, y, feat_names


def fit_and_eval_model(name, estimator, X_train, X_test, y_train, y_test, feat_names):
    """
    Fit pipeline (scaler + estimator), return dict of metrics and objects to save.
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    print(f"\nTraining {name} ...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # probabilities if available
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    clf = pipe.named_steps["clf"]
    feat_imp = None
    if hasattr(clf, "feature_importances_") and feat_names is not None:
        feat_imp = pd.DataFrame({
            "feature": feat_names,
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=False)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "name": name,
        "pipeline": pipe,
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importances": feat_imp
    }


def save_results(results_list, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    # summary CSV
    df_summary = pd.DataFrame([{"model": r["name"], "accuracy": r["accuracy"]} for r in results_list])
    df_summary.to_csv(os.path.join(out_dir, "baseline_results_summary.csv"), index=False)

    # save each model artifacts
    for r in results_list:
        model_tag = r["name"]
        # Save pipeline
        joblib.dump(r["pipeline"], os.path.join(out_dir, f"pipeline_{model_tag}.joblib"))
        # Save classification report (as CSV)
        df_report = pd.DataFrame(r["report"]).transpose()
        df_report.to_csv(os.path.join(out_dir, f"classification_report_{model_tag}.csv"))
        # Save confusion matrix
        cm_df = pd.DataFrame(r["confusion_matrix"])
        cm_df.to_csv(os.path.join(out_dir, f"confusion_matrix_{model_tag}.csv"), index=False)
        # Save predictions
        preds_df = pd.DataFrame({
            "y_test": r["y_test"],
            "y_pred": r["y_pred"]
        })
        if r["y_proba"] is not None:
            # save probabilities as additional columns
            proba_cols = [f"p_{i}" for i in range(r["y_proba"].shape[1])]
            proba_df = pd.DataFrame(r["y_proba"], columns=proba_cols)
            preds_df = pd.concat([preds_df.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
        preds_df.to_csv(os.path.join(out_dir, f"predictions_{model_tag}.csv"), index=False)
        # Save feature importances (if exists)
        if r["feature_importances"] is not None:
            r["feature_importances"].to_csv(os.path.join(out_dir, f"feature_importances_{model_tag}.csv"), index=False)

    # Save best model (highest accuracy)
    best = max(results_list, key=lambda x: x["accuracy"])
    joblib.dump(best["pipeline"], os.path.join(out_dir, f"best_pipeline_{best['name']}.joblib"))
    print("\nSaved summary and model artifacts to:", out_dir)


def main(prefix, test_size, out_dir):
    print("Loading feature dataset...")
    X, y, feat_names = load_data(prefix)
    print("Feature matrix:", X.shape)
    print("Labels:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "SVM_RBF": SVC(kernel="rbf", gamma="scale", C=1.0, probability=True),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42)
    }

    results = []
    for name, est in models.items():
        r = fit_and_eval_model(name, est, X_train, X_test, y_train, y_test, feat_names)
        results.append(r)

    save_results(results, out_dir, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="features/tracefinder", help="features prefix (no trailing suffix)")
    parser.add_argument("--test-size", default=0.2, type=float, help="test split fraction")
    parser.add_argument("--out-dir", default="models/baseline", help="output directory to save models and reports")
    args = parser.parse_args()

    main(args.prefix, args.test_size, args.out_dir)
