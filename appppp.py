import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="🔋 Solar Power Predictor", layout="centered")
st.title("⚡ Solar Power Generation Forecast")
st.markdown("Select real-world environmental data values to predict power generation (in Joules).")

# Load sample data for dropdowns
df = pd.read_csv("solarpowergeneration.csv")  # Make sure this file exists in your folder

# Load model
try:
    model, scaler, features = joblib.load("best_model.pkl")
    st.success("✅ Model loaded from best_model.pkl")
except:
    model = joblib.load("best_pipeline.joblib")
    scaler = None
    features = [
        'distance-to-solar-noon', 'temperature', 'wind-direction',
        'wind-speed', 'sky-cover', 'visibility', 'humidity',
        'average-wind-speed-(period)', 'average-pressure-(period)'
    ]
    st.success("✅ Model loaded from best_model.joblib")

# Show dropdowns with real values
st.header("📥 Select Input Values from Dataset")
inputs = []
for col in features:
    options = sorted(df[col].dropna().unique())
    val = st.selectbox(f"{col}", options)
    inputs.append(val)

# Predict
if st.button("🔮 Predict Power"):
    try:
        wind_speed = inputs[3]
        temperature = inputs[1]
        humidity = inputs[6]

        wind_power = wind_speed ** 2
        temp_humidity_ratio = temperature / (humidity + 1)

        final_inputs = inputs + [wind_power, temp_humidity_ratio]
        X = np.array(final_inputs).reshape(1, -1)

        if scaler:
            X = scaler.transform(X)

        prediction = model.predict(X)[0]
        st.success(f"🔋 Predicted Power Generation: **{prediction:.2f} Joules**")

        st.markdown("### 📌 Derived Features")
        st.write(f"**wind_power:** {wind_power:.2f}")
        st.write(f"**temp_humidity_ratio:** {temp_humidity_ratio:.4f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
