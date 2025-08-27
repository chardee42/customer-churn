from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")
model = joblib.load("artifacts/xgb_model.json")

class Customer(BaseModel):
    # include only a few representative features; you can expand:
    gender: str | None = None
    SeniorCitizen: int | None = None
    Partner: str | None = None
    Dependents: str | None = None
    tenure: float | None = None
    PhoneService: str | None = None
    Contract: str | None = None
    MonthlyCharges: float | None = None
    TotalCharges: float | None = None

@app.post("/predict")
def predict(cust: Customer):
    df = pd.DataFrame([cust.dict()])
    proba = model.predict_proba(df)[:,1][0]
    return {"churn_probability": float(proba)}
