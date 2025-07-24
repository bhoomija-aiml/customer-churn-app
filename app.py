import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Customer Churn Predictor")

st.title(" Customer Churn Prediction App")
st.write("Fill in the customer details below to predict if the customer is likely to churn.")

# --- Safe model loading with error handling ---
try:
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except Exception as e:
    st.error(f" Error loading model files: {e}")
    st.stop()  # Stop execution if files not found

# --- Input Fields ---
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has a Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

# --- Prepare User Input ---
user_input = {}

# Binary encoding
user_input['gender'] = 1 if gender == "Male" else 0
user_input['SeniorCitizen'] = 1 if senior == "Yes" else 0
user_input['Partner'] = 1 if partner == "Yes" else 0
user_input['Dependents'] = 1 if dependents == "Yes" else 0
user_input['tenure'] = tenure

# Add placeholders for one-hot encoded features (default 0)
for col in feature_columns:
    if col not in user_input:
        user_input[col] = 0

# Scale MonthlyCharges and TotalCharges
try:
    scaled_vals = scaler.transform([[monthly_charges, total_charges]])
    user_input['MonthlyCharges'] = scaled_vals[0][0]
    user_input['TotalCharges'] = scaled_vals[0][1]
except Exception as e:
    st.error(f" Error in scaling input: {e}")
    st.stop()

# Convert to proper input array
input_array = np.array([user_input[col] for col in feature_columns]).reshape(1, -1)

# --- Predict ---
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1] * 100

        if prediction == 1:
            st.error(f" This customer is likely to **churn**. (Probability: {probability:.2f}%)")
        else:
            st.success(f" This customer is **not likely to churn**. (Probability: {probability:.2f}%)")
    except Exception as e:
        st.error(f" Prediction failed: {e}")
