import streamlit as st
import pandas as pd
import requests

st.title("MaligNet: Breast Cancer Prediction")

# Upload CSV for Predictions
uploaded_file = st.file_uploader("Upload patient data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data", df.head())

    # Send data to backend API
    response = requests.post("http://127.0.0.1:5000/predict", json=df.to_dict(orient="records"))

    if response.status_code == 200:
        predictions = response.json()
        st.write("### Predictions", predictions)
    else:
        st.error("Failed to get predictions. Check API.")

