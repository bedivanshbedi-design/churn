import streamlit as st
import joblib
import pandas as pd
import os
from src.train import train_model   
from src.monitoring import generate_drift_report

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")

# Load model
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 60)
salary = st.sidebar.number_input("Salary", value=30000)
balance = st.sidebar.number_input("Balance", value=10000)
tenure = st.sidebar.slider("Tenure", 1, 10)

# -----------------------------
# 🔮 Prediction
# -----------------------------
if st.button("Predict"):
    data = pd.DataFrame([{
        "age": age,
        "salary": salary,
        "balance": balance,
        "tenure": tenure
    }])

    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]

    st.subheader("Result:")

    if pred == 1:
        st.error(f"Customer will churn ❌ (Prob: {proba:.2f})")
    else:
        st.success(f"Customer will stay ✅ (Prob: {proba:.2f})")


# =============================
# ➕ ADD DATA SECTION
# =============================
st.subheader("➕ Add New Data")

churn_label = st.selectbox("Actual Outcome (Churn)", [0, 1])

if st.button("Add Data"):
    new_data = pd.DataFrame([{
        "age": age,
        "salary": salary,
        "balance": balance,
        "tenure": tenure,
        "churn": churn_label
    }])

    file_path = "data/raw/data.csv"

    if os.path.exists(file_path):
        new_data.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(file_path, index=False)

    st.success("✅ New data added to dataset!")


# =============================
# 🔁 RETRAIN SECTION
# =============================
st.subheader("🔁 Retrain Model")

if st.button("Retrain Model"):
    with st.spinner("Training model... please wait ⏳"):
        model, acc = train_model()

    model = load_model()  # 👈 reload updated model
    st.success(f"✅ Model retrained! New Accuracy: {acc:.4f}")
    
    generate_drift_report()
# =============================
# 📂 UPLOAD CSV + AUTO RETRAIN
# =============================
st.subheader("📂 Upload New CSV Data")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)

    file_path = "data/raw/data.csv"

    # Append new dataset
    if os.path.exists(file_path):
        new_data.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(file_path, index=False)

    st.success("✅ CSV uploaded & data appended!")

    # 🔁 AUTO RETRAIN
    with st.spinner("Auto retraining model..."):
        model, acc = train_model()

    model = load_model()

    st.success(f"✅ Model retrained automatically! Accuracy: {acc:.4f}")

    # 📊 AUTO MONITORING
    with st.spinner("Updating monitoring report..."):
        generate_drift_report()

    st.success("📊 Monitoring report updated!")

# =============================
# 📊 MONITORING SECTION
# =============================
st.subheader("📊 Model Monitoring (Drift Detection)")

if st.button("Run Drift Monitoring"):
    with st.spinner("Generating drift report..."):
        generate_drift_report()
    st.success("✅ Drift report generated!")

# Display report if exists
if os.path.exists("drift_report.html"):
    st.subheader("📈 Drift Report")

    with open("drift_report.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=800, scrolling=True)