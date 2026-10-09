# 📊 Employee Attrition Analysis and Prediction

## 📌 Project Overview
Employee turnover poses a significant challenge for organizations, resulting in increased costs, reduced productivity, and team disruptions. This project delivers an end-to-end data science and machine learning pipeline to analyze employee records, identify primary churn vectors, and accurately predict at-risk talent using a predictive dashboard interface.

---

## 🛠️ Skills & Frameworks Acquired
* **Data Preprocessing & Structural Cleanup** (Pandas)
* **Exploratory Data Analysis (EDA)** (Feature correlation matrices)
* **Advanced Feature Engineering** (One-Hot Variable Encoding)
* **Machine Learning Pipeline Development** (Scikit-Learn Random Forest Architecture)
* **Synthetic Over-Sampling Execution** (SMOTE Data Imbalance Mitigation)
* **Interactive Full-Stack Web Deployment** (Streamlit Application Interface)

---

## 📈 System Core Pipeline Architecture
1. **Source Loading:** Reads 1,470 structural enterprise employee profiles.
2. **Data Cleaning:** Automatically purges zero-variance constant trackers (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`).
3. **Encoding:** Normalizes string matrices into dynamic mathematical parameters using One-Hot encoding.
4. **Resampling:** Utilizes SMOTE algorithmic configurations to fix 84/16 distribution imbalance ratios inside training data structures.
5. **Prediction Modeling:** Deploys a trained Random Forest Classification configuration yielding optimal precision bounds.

---

## 🚀 Local Deployment Execution Steps

To initialize this project workspace locally, execute the following parameters within your environment terminal:

```bash
# 1. Install necessary environment libraries
pip install pandas scikit-learn imbalanced-learn streamlit plotly joblib

# 2. Run the Machine Learning pipeline to train the model artifacts
python main.py

# 3. Launch the live interactive Streamlit Dashboard
streamlit run app.py
```
