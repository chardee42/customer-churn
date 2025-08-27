
# Customer Churn Prediction (XGBoost + Python)

Predicting which customers are likely to churn is critical for subscription-based businesses.  
This project uses **Python, Pandas, Scikit-learn, and XGBoost** to model churn on the Telco Customer dataset, achieving strong predictive performance.

---

## 🚀 Project Overview
- Built an **end-to-end ML pipeline**: data cleaning, feature engineering, model training, and evaluation.  
- Model: **XGBoost Classifier** with hyperparameter tuning via cross-validation.  
- Results: **AUC = 0.83, Accuracy = 79%** on hold-out test set.  
- Includes both an **API (FastAPI)** for programmatic access and a **Streamlit demo app** for non-technical users.

---

## 📂 Repository Structure
```

customer-churn/
├─ README.md
├─ requirements.txt
├─ data/                # dataset CSV (not pushed if large)
├─ notebooks/
│  └─ 01\_eda\_and\_model.ipynb   # Exploratory Data Analysis & baseline model
├─ src/
│  ├─ data\_prep.py      # data cleaning & feature engineering
│  ├─ train.py          # training script for XGBoost model
│  ├─ evaluate.py       # evaluation metrics, plots
│  └─ utils.py          # helper functions (logging, saving models, etc.)
├─ api/
│  └─ app.py            # FastAPI endpoint for predictions
├─ streamlit\_app.py     # Streamlit demo for interactive exploration
└─ .gitignore

````

---

## 🔑 Key Features
- **Exploratory Data Analysis (EDA):** visualized churn patterns across demographics and services.  
- **Feature Engineering:** encoded categorical features, imputed missing values, scaled numeric data.  
- **Modeling:** trained and tuned XGBoost classifier; compared with baseline models.  
- **Evaluation:** ROC curves, feature importance, confusion matrix.  
- **Deployment Options:**
  - 📊 **Streamlit app** for hands-on demo of predictions.  
  - ⚙️ **FastAPI endpoint** for programmatic use.  

---

## 🖼️ Demo
👉 **[Live Streamlit Demo](https://share.streamlit.io/YOUR-USERNAME/customer-churn/main/streamlit_app.py)**  
*(replace with your link once deployed)*

Sample screenshot:  
![Streamlit Demo Screenshot](docs/demo_screenshot.png)

---

## ⚙️ Installation & Usage
1. Clone this repo:
   ```bash
   git clone https://github.com/YOUR-USERNAME/customer-churn.git
   cd customer-churn


2. Create environment & install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter notebook for EDA & training:

   ```bash
   jupyter notebook notebooks/01_eda_and_model.ipynb
   ```
4. Launch Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```
5. Or run FastAPI locally:

   ```bash
   uvicorn api.app:app --reload
   ```

---

## 📊 Results

* **AUC:** 0.83
* **Accuracy:** 79%
* **Top Features:** contract type, tenure, internet service, monthly charges.

---

## 🛠️ Tech Stack

* **Languages:** Python (Pandas, NumPy, Scikit-learn, XGBoost)
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Deployment:** Streamlit, FastAPI, Uvicorn
* **Environment:** Jupyter, pip/venv

---

## ✨ Next Steps

* Improve feature engineering (interaction terms, customer lifetime value).
* Experiment with ensemble methods (Stacked models).
* Deploy API + Streamlit to cloud (Render, Railway, or Streamlit Cloud).

---

## 📌 Project Links

* 📂 **GitHub Repo:** [Customer Churn Prediction](https://github.com/YOUR-USERNAME/customer-churn)
* 📊 **Live Demo:** [Streamlit App](https://share.streamlit.io/YOUR-USERNAME/customer-churn/main/streamlit_app.py)

---

### Author

**Chris Hardee**
📍 Winter Park, FL
🔗 [LinkedIn](https://linkedin.com/in/chrishardeedataengineer) | [GitHub](https://github.com/YOUR-USERNAME)

```
