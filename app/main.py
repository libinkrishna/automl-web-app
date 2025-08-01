# app/main.py

import streamlit as st
import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automl import run_automl
from utils import download_link

st.set_page_config(page_title="AutoML Web App", layout="wide")
st.title("🤖 AutoML Web App")
st.write("Upload your CSV dataset, and let AI pick the best model for you!")

uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])
target_column = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Dataset")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    selectable_columns = [col for col in columns if col.lower() != "id"]
    target_column = st.selectbox("🎯 Select Target Column", selectable_columns)

    if st.button("🚀 Run AutoML"):
        with st.spinner("Running AutoML..."):
            model, report, predictions = run_automl(df, target_column)

        st.success("✅ Model training complete!")
        st.subheader("📈 Model Performance")
        st.code(report)

        st.subheader("📥 Download Predictions")
        st.dataframe(predictions.head())
        st.markdown(download_link(predictions, "predictions.csv", "Download CSV"), unsafe_allow_html=True)
