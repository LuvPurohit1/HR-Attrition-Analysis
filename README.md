# 👥 HR Employee Attrition Analysis & Prediction

> An end-to-end Data Science project analyzing employee attrition patterns and predicting which employees are likely to leave, using **Python**, **Random Forest ML Model**, and **Power BI**.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hr-attrition-analysis-j5yuzyykuigj7mnuqcjehm.streamlit.app/)

---

## 📊 Key Results

| Metric | Result |
|---|---|
| 🤖 ML Model Accuracy | **84.35%** |
| 📉 Overall Attrition Rate | **16.12%** |
| 💰 Avg Income — Employees Who Left | **$4,787.09/month** |
| 💰 Avg Income — Employees Who Stayed | **$6,832.74/month** |
| ⏰ Attrition WITH Overtime | **30.53%** |
| ✅ Attrition WITHOUT Overtime | **10.44%** |
| 🏆 Top Attrition Factor | **Monthly Income** |
| 📦 Dataset Size | **1,470 employees** |

---

## 🎯 Project Objective

Analyze IBM HR data to:
- Identify key factors driving employee attrition
- Predict which employees are at risk of leaving
- Provide actionable insights for HR decision-making
- Build an ML model to flag high-risk employees

---

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 📂 Dataset

**Source:** [IBM HR Analytics Employee Attrition Dataset — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
**File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`
**Size:** 1,470 employees | 35 features

---

## 📁 Project Structure

```
HR-Attrition-Analysis/
│
├── Data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── Notebook/
│   └── hr_attrition_analysis.ipynb
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🔍 Exploratory Data Analysis (EDA)

- **Attrition Rate** — 16.12% of employees left the company
- **Department Analysis** — Sales has highest attrition rate
- **Age Group** — Employees aged 18-25 have highest attrition (youngest group most at risk)
- **Income Impact** — Employees who left earned $2,045 less per month on average
- **Overtime Effect** — Overtime employees are **3x more likely** to leave (30.53% vs 10.44%)
- **Work-Life Balance** — Lower work-life balance scores directly linked to higher attrition

---

## 🤖 Machine Learning — Attrition Prediction

Built a **Random Forest Classifier** to predict whether an employee will leave.

**ML Workflow:**
1. Data Cleaning & Preprocessing
2. Label Encoding of categorical variables
3. Train-Test Split (80/20, stratified)
4. Random Forest Model Training (100 estimators)
5. Evaluation — Accuracy, Classification Report, Confusion Matrix
6. Feature Importance Analysis

**Result:** **84.35% Accuracy** — model correctly identifies 84 out of 100 at-risk employees

---

## 💡 Key Business Insights

- 💰 **Monthly Income is the #1 factor** — employees earning less are significantly more likely to leave
- ⏰ **Overtime triples attrition risk** — 30.53% vs 10.44% without overtime
- 👶 **Young employees (18-25) are most at risk** — need better retention programs for fresh talent
- 📉 **Income gap of $2,045/month** between those who left vs stayed — a clear retention lever
- 🏢 **Sales department** needs urgent attention — highest attrition rate across departments

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/LuvPurohit1/HR-Attrition-Analysis.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook Notebook/hr_attrition_analysis.ipynb

# 4. Run the Streamlit app
streamlit run app.py
```

---

## 🚀 Future Improvements

- [ ] Try XGBoost and compare accuracy
- [ ] Add SHAP values for model explainability
- [ ] Build a department-wise attrition dashboard in Power BI
- [ ] Add salary recommendation engine for retention

---

## 👤 Author

**Luv Purohit**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/luv-purohit-40693634a)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/LuvPurohit1)
