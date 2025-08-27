# Customer Churn Prediction (Telecom/SaaS)

Predicts which customers are likely to cancel service using **Python, Scikit-learn, XGBoost, and SHAP**. Includes a simple **FastAPI** endpoint and optional **Streamlit** demo.

## Dataset
Telco Customer Churn (public Kaggle dataset). Place CSV in `data/` as `telco_churn.csv`.  
*Note: no proprietary data used.*

## Tech
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, SHAP
- Matplotlib/Seaborn
- (Optional) FastAPI + Uvicorn, Streamlit

## Quickstart
```bash
pip install -r requirements.txt
python -m src.train --csv_path data/telco_churn.csv --model_path artifacts/xgb_model.json
python -m src.evaluate --csv_path data/telco_churn.csv --model_path artifacts/xgb_model.json
uvicorn api.app:app --reload  # optional API
streamlit run streamlit_app.py # optional demo
