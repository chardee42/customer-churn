import argparse, json
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report
from .data_prep import load_data, split_X_y

def main(csv_path: str, model_path: str):
    model = joblib.load(model_path)
    df = load_data(csv_path)
    X, y = split_X_y(df)
    proba = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, proba)
    print(f"Holdout ROC-AUC: {auc:.3f}")
    # Optional: choose threshold
    y_pred = (proba >= 0.5).astype(int)
    print(classification_report(y, y_pred))

    with open("artifacts/eval.json","w") as f:
        json.dump({"roc_auc": float(auc)}, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--model_path", default="artifacts/xgb_model.json")
    args = ap.parse_args()
    main(args.csv_path, args.model_path)
