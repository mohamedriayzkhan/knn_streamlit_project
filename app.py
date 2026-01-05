import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("🩺 Breast Cancer Prediction (KNN Model)")

st.write("Enter tumor measurements to predict diagnosis")

# Load model
model = joblib.load("model.pkl")

# Load dataset to get feature names
df = pd.read_csv("data/dataset.csv")
df.drop(columns=["id", "Unnamed: 32", "diagnosis"], inplace=True)

# Input fields
input_data = {}
for col in df.columns:
    input_data[col] = st.number_input(col, value=float(df[col].mean()))

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result = "Malignant (Cancer)" if prediction == 1 else "Benign (No Cancer)"

    st.subheader("🔍 Prediction Result")
    st.success(result)
    st.write(f"Confidence: {max(probability)*100:.2f}%")
