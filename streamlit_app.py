import streamlit as st, joblib, pandas as pd
model = joblib.load("artifacts/xgb_model.json")

st.title("Customer Churn Predictor")
tenure = st.slider("Tenure (months)", 0, 72, 6)
monthly = st.number_input("MonthlyCharges", min_value=0.0, value=75.0)
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
gender = st.selectbox("Gender", ["Male","Female"])
total = st.number_input("TotalCharges", min_value=0.0, value=500.0)

if st.button("Predict"):
    df = pd.DataFrame([{
        "tenure": tenure, "MonthlyCharges": monthly, "Contract": contract,
        "gender": gender, "TotalCharges": total
    }])
    proba = model.predict_proba(df)[:,1][0]
    st.metric("Churn Probability", f"{proba:.2%}")
