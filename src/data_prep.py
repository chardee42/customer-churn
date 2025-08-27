
---

# `src/data_prep.py`

```python
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

TARGET = "Churn"  # assumes 'Yes'/'No'

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleanup example:
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def split_X_y(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = (df[TARGET].astype(str).str.lower() == "yes").astype(int)
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["int64","float64"]).columns.tolist()
    num_pipe = SimpleImputer(strategy="median")
    cat_pipe = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
    )
    return pre
