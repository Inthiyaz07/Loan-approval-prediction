import os
import sys
import subprocess
import pickle
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(_file_))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")

# Train model if it does not exist
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training model...")
    subprocess.run([sys.executable, "train_model.py"], check=True)

# Load model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("Model loading failed")
    st.code(str(e))
    st.stop()

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


