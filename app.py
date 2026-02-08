import streamlit as st
import pickle
import numpy as np
import os

# Page config
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")

st.title("💰 Loan Approval Prediction")
st.write("Enter applicant details to predict loan approval")

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found: loan_model.pkl")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
applicant_income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", [1, 0])

# Encode inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0

input_data = np.array([[gender, married, education, applicant_income, loan_amount, credit_history]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
