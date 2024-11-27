import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from app.encoder_loader import load_encoder
from app.model_loader import load_model
from app.scaler_loader import load_scaler

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load the model, encoder, and scaler
model = load_model()
scaler = load_scaler()
encoder = load_encoder()

if model is None or scaler is None or encoder is None:
    st.error("Model, scaler, or encoder could not be loaded.")
    st.stop()

# Streamlit app title
st.title("Churn Prediction Application by JozarRios")

# Sidebar for user input
st.sidebar.header("Customer Data Input")
credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 0.0)
num_of_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.sidebar.selectbox("Is the Customer has a Credit Card", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
is_active_member = st.sidebar.selectbox("Is the Customer an active member", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Prepare the input data
input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}])

# Preprocess input data
input_data["Gender"] = encoder.transform(input_data["Gender"])
input_data = pd.get_dummies(input_data, columns=["Geography"], drop_first=True)

# Add missing columns
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the training data
input_data = input_data[model.feature_names_in_]

# Scale numerical features
numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Prediction
prediction = model.predict(input_data)[0]
prediction_prob = model.predict_proba(input_data)[:, 1][0]

# Display results
st.subheader("Prediction Result")
if prediction == 1:
    st.warning(f"The customer is **likely to churn** (Probability: {prediction_prob:.2f}).")
else:
    st.success(f"The customer is **not likely to churn** (Probability: {prediction_prob:.2f}).")

# Feature importance bar chart
st.subheader("Feature Importance Affecting Prediction")
feature_importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
top_features = feature_importance.sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
sns.barplot(x=top_features, y=top_features.index, ax=ax)
ax.set_title("Top 10 Features Affecting Prediction")
ax.set_xlabel("Importance")
st.pyplot(fig)

# Probability distribution chart
st.subheader("Prediction Probability Distribution")
fig, ax = plt.subplots()
sns.barplot(x=["Not Churn", "Churn"], y=[1 - prediction_prob, prediction_prob], ax=ax)
ax.set_title("Prediction Probability")
ax.set_ylabel("Probability")
st.pyplot(fig)
