import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")

# Load model
model = joblib.load("model.pkl")

# Sidebar inputs
st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider("Age", 18, 60)
salary = st.sidebar.number_input("Salary", value=30000)
balance = st.sidebar.number_input("Balance", value=10000)
tenure = st.sidebar.slider("Tenure", 1, 10)

# Prediction
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