import streamlit as st

st.title("🤖 AutoML Web App")
st.write("Upload a CSV file and let AI do the rest!")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    st.success("File uploaded! AutoML coming soon...")
