import argparse, os, json
import joblib
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .data_prep import load_data, split_X_y, build_preprocessor

def main(csv_path: str, model_path: str):
    os.makedirs("artifacts", exist_ok=True)
    df = load_data(csv_path)
    X, y = split_X_y(df)
    pre = build_preprocessor(X)

    models = {
        "logreg": (LogisticRegression(max_iter=1000), {"clf__C":[0.1,1,3]}),
        "rf": (RandomForestClassifier(n_estimators=300, n_jobs=-1), {"clf__max_depth":[None,8,16]}),
        "xgb": (XGBClassifier(
                    n_estimators=500, eval_metric="auc", tree_method="hist", n_jobs=-1),
                {"clf__max_depth":[4,6,8], "clf__learning_rate":[0.03,0.1], "clf__subsample":[0.8,1.0]})
    }

    best_auc = -1
    best_pipe = None
    best_name = None

    for name, (clf, grid) in models.items():
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1)
        gs.fit(X, y)
        if gs.best_score_ > best_auc:
            best_auc = gs.best_score_
            best_pipe = gs.best_estimator_
            best_name = name

    joblib.dump(best_pipe, model_path)
    with open("artifacts/metrics.json","w") as f:
        json.dump({"best_model": best_name, "cv_roc_auc": best_auc}, f, indent=2)
    print(f"Saved {best_name} to {model_path} with CV ROC-AUC={best_auc:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--model_path", default="artifacts/xgb_model.json")
    args = ap.parse_args()
    main(args.csv_path, args.model_path)
