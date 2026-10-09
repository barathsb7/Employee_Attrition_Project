import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="HR Analytics Portal", layout="wide", page_icon="📊")

# 2. Load trained machine learning model artifacts safely
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('attrition_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except Exception as e:
        st.error("Error loading model files. Please ensure main.py has run successfully first.")
        return None, None

model, model_columns = load_model_artifacts()

# 3. Sidebar Navigation Control
st.sidebar.title("🎛️ Navigation Portal")
app_mode = st.sidebar.radio("Go To:", ["Dashboard Overview", "Predict Attrition Risk"])

# 4. Mode 1: Dashboard Analytics View
if app_mode == "Dashboard Overview":
    st.title("📊 Employee Attrition Analytics Dashboard")
    st.markdown("---")
    
    # Load dataset to display trends
    @st.cache_data
    def load_data():
        return pd.read_csv("employee_attrition.csv")
    
    df_raw = load_data()
    
    # Top KPI Metrics row
    total_emp = len(df_raw)
    attrition_rate = (df_raw['Attrition'].value_counts(normalize=True).get('Yes', 0)) * 100
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Workforce Headcount", f"{total_emp} Employees")
    kpi2.metric("Current Attrition Rate", f"{attrition_rate:.1f}%")
    kpi3.metric("Critical Retention Risk Drivers", "Overtime, Income")
    
    st.markdown("### 📋 Enterprise Workforce Records Snapshot")
    st.dataframe(df_raw.head(20), use_container_width=True)

# 5. Mode 2: Individual Risk Prediction Calculator
elif app_mode == "Predict Attrition Risk" and model is not None:
    st.title("🔮 Employee Attrition Predictive Tool")
    st.markdown("Provide employee details below to analyze individual voluntary turnover probability matrix.")
    
    with st.form("input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 65, 35)
            monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000)
            overtime = st.selectbox("Working Overtime?", ["Yes", "No"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            
        with col2:
            job_level = st.slider("Job Seniority Level", 1, 5, 2)
            total_years = st.slider("Total Working Years Career-wide", 0, 40, 10)
            years_at_company = st.slider("Years At Current Company", 0, 40, 5)
            years_in_role = st.slider("Years In Current Active Role", 0, 20, 3)
            
        with col3:
            job_sat = st.slider("Job Satisfaction Rating (1-4)", 1, 4, 3)
            env_sat = st.slider("Environment Satisfaction Rating (1-4)", 1, 4, 3)
            work_life = st.slider("Work-Life Balance Rating (1-4)", 1, 4, 3)
            travel = st.selectbox("Business Travel Frequency", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            
        submit = st.form_submit_button("Run Risk Prediction Analysis")
        
        if submit:
            # Fixed index location matrix layout logic placeholder
            input_data = pd.DataFrame(0, index=[0], columns=model_columns)
            
            # Map input features
            input_data["Age"] = age
            input_data["MonthlyIncome"] = monthly_income
            input_data["JobLevel"] = job_level
            input_data["TotalWorkingYears"] = total_years
            input_data["YearsAtCompany"] = years_at_company
            input_data["YearsInCurrentRole"] = years_in_role
            input_data["JobSatisfaction"] = job_sat
            input_data["EnvironmentSatisfaction"] = env_sat
            input_data["WorkLifeBalance"] = work_life
            
            # Explicitly match one-hot encoded flags safely
            if overtime == "Yes": input_data["OverTime_Yes"] = 1
            if travel == "Travel_Frequently": input_data["BusinessTravel_Travel_Frequently"] = 1
            elif travel == "Travel_Rarely": input_data["BusinessTravel_Travel_Rarely"] = 1
            if marital_status == "Married": input_data["MaritalStatus_Married"] = 1
            elif marital_status == "Single": input_data["MaritalStatus_Single"] = 1
            
            # Generate classification outcomes
            probability = model.predict_proba(input_data)[0][1] * 100
            
            st.markdown("---")
            if probability > 50:
                st.error(f"⚠️ **High Retention Risk Warning Alert:** This employee has a **{probability:.1f}% probability** of voluntary departure. Immediate proactive engagement or review recommended.")
            else:
                st.success(f"✅ **Stable Retention Status:** This employee shows a **{probability:.1f}% risk score**. Standard engagement baseline parameters met.")
