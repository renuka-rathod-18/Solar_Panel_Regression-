import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler (if exists)
try:
    model, scaler, feature_names = joblib.load("best_model.pkl")  # for .pkl
except:
    model = joblib.load("best_pipeline.joblib")  # for .joblib
    scaler = None
    feature_names = ['distance_to_solar_noon', 'temperature', 'wind_direction',
                     'wind_speed', 'sky_cover', 'visibility', 'humidity',
                     'average_wind_speed', 'average_pressure', 'wind_power', 'temp_humidity_ratio']

st.set_page_config(page_title="🔋 Solar Power Predictor", layout="centered")
st.title("⚡ Solar Power Generation Forecast")
st.markdown("Enter values for the environmental parameters to predict solar power generation (in Joules).")

# User input for each feature
inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    inputs.append(value)

# Predict
if st.button("🔮 Predict Power"):
    try:
        input_array = np.array(inputs).reshape(1, -1)

        # Apply scaling if required
        if scaler:
            input_array = scaler.transform(input_array)

        # Predict
        predicted_power = model.predict(input_array)[0]
        st.success(f"Predicted Power Generation: **{predicted_power:.2f} Joules**")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
