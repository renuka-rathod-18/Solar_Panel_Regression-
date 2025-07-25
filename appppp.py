import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="🔋 Solar Power Generation Predictor", layout="centered")
st.title("⚡ Solar Power Generation Forecast")
st.markdown("Enter values for the environmental parameters to predict solar power generation (in Joules).")

# Try loading either a .pkl (with scaler) or a .joblib (without scaler)
try:
    model, scaler, features = joblib.load("best_model.pkl")
    st.success("✅ Model loaded from best_model.pkl")
except:
    model = joblib.load("best_pipeline.joblib")
    scaler = None
    features = ['distance_to_solar_noon', 'temperature', 'wind_direction',
                'wind_speed', 'sky_cover', 'visibility', 'humidity',
                'average_wind_speed', 'average_pressure']
    st.success("✅ Model loaded from best_model.joblib")

# Input fields using text_input (no stepper buttons)
st.header("📥 Input Features")
user_inputs = []
for feature in features:
    val = st.text_input(f"Enter {feature}:", value="0.0")
    try:
        val = float(val)
    except ValueError:
        st.warning(f"⚠️ {feature} must be a number. Using default 0.0.")
        val = 0.0
    user_inputs.append(val)

# Predict Button
if st.button("🔮 Predict Power"):
    try:
        # Derived features if model requires them
        if len(features) == 10:  # joblib case
            wind_speed = user_inputs[3]
            temperature = user_inputs[1]
            humidity = user_inputs[6]

            wind_power = wind_speed ** 2
            temp_humidity_ratio = temperature / (humidity + 1)

            user_inputs.extend([wind_power, temp_humidity_ratio])

        # Convert to array
        X = np.array(user_inputs).reshape(1, -1)

        # Apply scaling if needed
        if scaler:
            X = scaler.transform(X)

        # Predict
        predicted = model.predict(X)[0]
        st.success(f"🔋 Predicted Power Generation: **{predicted:.2f} Joules**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
