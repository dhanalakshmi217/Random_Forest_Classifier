import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load pickle files
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title("🔮 ML Prediction App")

st.write("Enter input values")

# -------------------------------
# User input section
# -------------------------------
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0
    )

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    st.success(f"✅ Prediction Result: {prediction[0]}")
