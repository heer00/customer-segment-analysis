import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/kmeans_model.pkl")
scaler = joblib.load("model/scaler.pkl")
st.title("Customer Segmentation App")
st.write("Enter customer details to predict segment.")
income = st.number_input("Annual Income (k$)", min_value=0)
spending = st.number_input("Spending Score (1-100)", min_value=0, max_value=100)

cluster_labels = {
    0: "Premium Customers (High Income, High Spending)",
    1: "Careful Rich (High Income, Low Spending)",
    2: "Budget Spenders (Low Income, High Spending)",
    3: "Low Value Customers (Low Income, Low Spending)",
    4: "Average Customers (Medium Income, Medium Spending)"
}

if st.button("Predict Segment"):
    data = np.array([[income, spending]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]

    st.success(f"Segment: {cluster_labels[prediction]}")