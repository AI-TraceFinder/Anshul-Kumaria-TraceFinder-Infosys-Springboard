# src/train_optimized.py
"""
Optimized training:
- RandomizedSearchCV for RandomForest and SVM
- Optionally XGBoost (if installed)
- StackingClassifier combining best learners
- Saves pipelines, reports, confusion matrices, summary CSV

"""
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform

# Try to import xgboost (optional)
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

def load_data(prefix):
    X = np.load(prefix + "_features.npy")
    y = np.load(prefix + "_labels.npy")
    # feature names if present
    feat_names = None
    csv_path = prefix + "_features.csv"
    if os.path.exists(csv_path):
        try:
            dfh = pd.read_csv(csv_path, nrows=0)
            feat_names = list(dfh.columns[3:])
        except Exception:
            feat_names = None
    return X, y, feat_names

def run_randomized_search(clf, param_distributions, X_train, y_train, cv, n_iter=30, random_state=42):
    rs = RandomizedSearchCV(clf, param_distributions=param_distributions, n_iter=n_iter,
                            scoring="accuracy", cv=cv, n_jobs=-1, verbose=1, random_state=random_state)
    t0 = time()
    rs.fit(X_train, y_train)
    print("RandomizedSearch done in {:.1f}s — best score {:.4f}".format(time() - t0, rs.best_score_))
    return rs

def fit_and_save_pipeline(estimator, X_train, y_train, X_test, y_test, name, out_dir, feat_names=None):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    t0 = time()
    pipe.fit(X_train, y_train)
    train_time = time() - t0

    y_pred = pipe.predict(X_test)
    y_proba = None
    try:
        y_proba = pipe.predict_proba(X_test)
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(out_dir, f"pipeline_{name}.joblib"))
    with open(os.path.join(out_dir, f"best_params_{name}.json"), "w") as f:
        json.dump(getattr(estimator, "get_params", lambda: {})(), f, indent=2)
    pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, f"classification_report_{name}.csv"))
    pd.DataFrame(cm).to_csv(os.path.join(out_dir, f"confusion_matrix_{name}.csv"), index=False)

    preds_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
    if y_proba is not None:
        proba_cols = [f"p_{i}" for i in range(y_proba.shape[1])]
        preds_df = pd.concat([preds_df.reset_index(drop=True), pd.DataFrame(y_proba, columns=proba_cols)], axis=1)
    preds_df.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"), index=False)

    # feature importances for RF
    clf_only = pipe.named_steps["clf"]
    if hasattr(clf_only, "feature_importances_") and feat_names is not None:
        imp = pd.DataFrame({"feature": feat_names, "importance": clf_only.feature_importances_}).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(out_dir, f"feature_importances_{name}.csv"), index=False)

    summary = {"model": name, "accuracy": acc, "train_time_sec": train_time}
    return summary

def main(features_prefix, out_dir, cv_folds, n_iter, test_size, random_state):
    print("Loading data...")
    X, y, feat_names = load_data(features_prefix)
    print("X:", X.shape, "y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # RandomForest search space
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_params = {
        "n_estimators": randint(200, 1001),
        "max_depth": randint(6, 41),
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": randint(2, 11),
        "class_weight": [None, "balanced"]
    }

    # SVM search (on raw SVC, we'll wrap with scaler later)
    svm = SVC(kernel="rbf", probability=True, random_state=random_state)
    svm_params = {
        "C": uniform(0.1, 10),
        "gamma": ["scale", "auto"]  # you can also try float values
    }

    print("Running RandomizedSearch for RandomForest...")
    rs_rf = run_randomized_search(rf, rf_params, X_train, y_train, cv=cv, n_iter=n_iter, random_state=random_state)

    print("Running RandomizedSearch for SVM (this may be slower)...")
    rs_svm = run_randomized_search(svm, svm_params, X_train, y_train, cv=cv, n_iter=max(8, n_iter//5), random_state=random_state)

    best_rf = rs_rf.best_estimator_
    best_svm = rs_svm.best_estimator_

    # Optionally XGBoost
    best_xgb = None
    if HAVE_XGB:
        print("XGBoost detected — will include in stacking (no search).")
        best_xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1, random_state=random_state)
        best_xgb.fit(X_train, y_train)

    # Save the best params
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "random_search_best_params.json"), "w") as f:
        json.dump({"rf": rs_rf.best_params_, "svm": rs_svm.best_params_}, f, indent=2)

    # Fit final pipelines and save artifacts
    summaries = []
    summaries.append(fit_and_save_pipeline(best_rf, X_train, y_train, X_test, y_test, "RandomForest", out_dir, feat_names))
    summaries.append(fit_and_save_pipeline(best_svm, X_train, y_train, X_test, y_test, "SVM_RBF", out_dir, feat_names))

    if HAVE_XGB:
        # save XGB pipeline
        summaries.append(fit_and_save_pipeline(best_xgb, X_train, y_train, X_test, y_test, "XGB", out_dir, feat_names))

    # Build stacking classifier (use pipelines that include scaler inside stacking)
    estimators = [("rf", best_rf), ("svm", best_svm)]
    if best_xgb is not None:
        estimators.append(("xgb", best_xgb))

    meta = LogisticRegression(max_iter=500, random_state=random_state)
    stacking = StackingClassifier(estimators=estimators, final_estimator=meta, n_jobs=-1, passthrough=False)
    print("Training stacking classifier...")
    # wrap stacking in pipeline with scaler
    stack_pipe = Pipeline([("scaler", StandardScaler()), ("stack", stacking)])
    t0 = time()
    stack_pipe.fit(X_train, y_train)
    print("Stacking trained in {:.1f}s".format(time() - t0))

    # evaluate stacking
    y_pred_stack = stack_pipe.predict(X_test)
    acc_stack = accuracy_score(y_test, y_pred_stack)
    print("Stacking accuracy:", acc_stack)
    # save stacking artifacts
    joblib.dump(stack_pipe, os.path.join(out_dir, "pipeline_stacking.joblib"))
    pd.DataFrame(classification_report(y_test, y_pred_stack, output_dict=True)).transpose().to_csv(os.path.join(out_dir, "classification_report_stacking.csv"))
    pd.DataFrame(confusion_matrix(y_test, y_pred_stack)).to_csv(os.path.join(out_dir, "confusion_matrix_stacking.csv"), index=False)
    pd.DataFrame({"model": ["stacking"], "accuracy": [acc_stack]}).to_csv(os.path.join(out_dir, "stacking_summary.csv"), index=False)

    # summary of all
    df_summary = pd.concat([pd.DataFrame(s, index=[0]) for s in summaries], ignore_index=True, sort=False)
    # Add stacking row
    df_summary = pd.concat([df_summary, pd.DataFrame([{"model": "stacking", "accuracy": acc_stack}])], ignore_index=True, sort=False)
    df_summary.to_csv(os.path.join(out_dir, "final_summary.csv"), index=False)
    print("Saved all artifacts to", out_dir)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-prefix", default="features/tracefinder_v2")
    parser.add_argument("--out-dir", default="models/baseline_v2")
    parser.add_argument("--cv", default=5, type=int)
    parser.add_argument("--n-iter", default=40, type=int, help="n_iter for RandomizedSearch")
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--random-state", default=42, type=int)
    args = parser.parse_args()

    main(args.features_prefix, args.out_dir, args.cv, args.n_iter, args.test_size, args.random_state)
