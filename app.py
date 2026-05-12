import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="centered"
)

# ── Load & Train Model ────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df_ml = df.copy()

    df_ml.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'],
               axis=1, inplace=True)

    le_dict = {}
    categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        le_dict[col] = le

    X = df_ml.drop('Attrition', axis=1)
    y = df_ml['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns.tolist(), df, le_dict

model, feature_cols, df, le_dict = load_and_train()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("👥 HR Employee Attrition Predictor")
st.markdown("**Enter employee details to predict if they are at risk of leaving the company.**")
st.markdown("---")

# ── Key Stats ─────────────────────────────────────────────────────────────────
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", "1,470")
col2.metric("Attrition Rate", "16.12%")
col3.metric("Model Accuracy", "84.35%")
col4.metric("Top Risk Factor", "Income")

st.markdown("---")

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("🔍 Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500)
    overtime = st.selectbox("Overtime", ["Yes", "No"])
    job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
    work_life_balance = st.slider("Work-Life Balance (1=Low, 4=High)", 1, 4, 3)

with col2:
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    distance_from_home = st.slider("Distance from Home (km)", 1, 30, 10)
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    education = st.slider("Education Level (1-5)", 1, 5, 3)
    num_companies = st.slider("Number of Companies Worked", 0, 9, 2)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🎯 Predict Attrition Risk", use_container_width=True):

    dept_map = {"Sales": 2, "Research & Development": 1, "Human Resources": 0}
    overtime_map = {"Yes": 1, "No": 0}

    # Build input with all required features using dataset medians as defaults
    input_data = {}
    for col in feature_cols:
        if df[col].dtype == 'object':
            input_data[col] = 0
        else:
            input_data[col] = int(df[col].median())

    # Override with user inputs
    input_data['Age'] = age
    input_data['MonthlyIncome'] = monthly_income
    input_data['OverTime'] = overtime_map[overtime]
    input_data['JobSatisfaction'] = job_satisfaction
    input_data['WorkLifeBalance'] = work_life_balance
    input_data['YearsAtCompany'] = years_at_company
    input_data['DistanceFromHome'] = distance_from_home
    input_data['Department'] = dept_map[department]
    input_data['Education'] = education
    input_data['NumCompaniesWorked'] = num_companies

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    attrition_prob = probability[1] * 100
    stay_prob = probability[0] * 100

    st.markdown("---")
    st.subheader("📋 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK — This employee is likely to leave ({attrition_prob:.1f}% probability)")
    else:
        st.success(f"✅ LOW RISK — This employee is likely to stay ({stay_prob:.1f}% probability)")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk of Leaving", f"{attrition_prob:.1f}%",
                  delta="High Risk" if attrition_prob > 50 else "Low Risk")
    with col2:
        st.metric("Likely to Stay", f"{stay_prob:.1f}%")

    # ── Risk Gauge Chart ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3))
    color = '#e74c3c' if attrition_prob > 50 else '#2ecc71'
    ax.barh(['Attrition Risk'], [attrition_prob], color=color, height=0.4)
    ax.barh(['Attrition Risk'], [100 - attrition_prob],
            left=[attrition_prob], color='#ecf0f1', height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Attrition Risk Score', fontweight='bold')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ── HR Recommendations ────────────────────────────────────────────────────
    st.subheader("💡 HR Recommendations")

    if monthly_income < 4787:
        st.warning("💰 **Salary below average of employees who left ($4,787)** — Consider a salary review")
    if overtime == "Yes":
        st.warning("⏰ **Overtime increases attrition risk by 3x** — Review workload distribution")
    if job_satisfaction <= 2:
        st.warning("😔 **Low job satisfaction** — Schedule a 1:1 feedback session")
    if work_life_balance <= 2:
        st.warning("⚖️ **Poor work-life balance** — Consider flexible working options")
    if years_at_company < 2:
        st.info("👶 **New employee** — Ensure proper onboarding and mentorship program")

    if attrition_prob <= 30:
        st.success("✅ Employee is well-retained. Keep up current engagement practices!")

# ── Key Insights ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💡 Key Insights from Analysis")

col1, col2 = st.columns(2)
with col1:
    st.error("⏰ **Overtime = 3x Attrition Risk**\n\n30.53% vs 10.44% without overtime")
    st.warning("💰 **Income Gap = $2,045/month**\n\nLeavers earn significantly less")
with col2:
    st.info("🏆 **Top Factor: Monthly Income**\n\nBiggest driver of attrition")
    st.success("🤖 **Model Accuracy: 84.35%**\n\nRandom Forest Classifier")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Made by **Luv Purohit** | "
    "[GitHub](https://github.com/LuvPurohit1) | "
    "[LinkedIn](https://www.linkedin.com/in/luv-purohit-40693634a)"
)
