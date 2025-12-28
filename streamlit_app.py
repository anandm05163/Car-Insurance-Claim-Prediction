import streamlit as st
import requests

st.title("Model Prediction API")

st.write("Click on a model to see predictions and metrics:")

# Function to fetch and display model metrics from Flask API
def fetch_metrics(model_name):
    response = requests.get(f'http://localhost:5000/predict/{model_name}')
    if response.status_code == 200:
        data = response.json()
        st.write(f"### {data['model'].capitalize()} Metrics")
        st.write(f"**Accuracy:** {data['metrics']['accuracy']:.2f}")
        st.write(f"**Precision:** {data['metrics']['precision']:.2f}")
        st.write(f"**Recall:** {data['metrics']['recall']:.2f}")
        st.write(f"**F1 Score:** {data['metrics']['f1_score']:.2f}")
    else:
        st.write("Error fetching data from the server.")

# Buttons for model predictions
if st.button('Random Forest'):
    fetch_metrics('random_forest')

if st.button('XGBoost'):
    fetch_metrics('xgboost')

if st.button('LightGBM'):
    fetch_metrics('lightgbm')
