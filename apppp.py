from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = load_model("OneDrive/Desktop/Project3/Power-Supply/power_model.h5")
scaler = joblib.load("OneDrive/Desktop/Project3/Power-Supply/scaler.pkl")

# Feature names used during training
FEATURES = ['hour', 'day', 'weekday', 'month']

@app.route('/')
def home():
    return "Power Supply Forecast API. Use POST /predict with JSON {\"datetime\": \"YYYY-MM-DD HH:MM:SS\"}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read datetime from request
        data = request.get_json()
        dt = pd.to_datetime(data['datetime'])

        # Extract features
        features = pd.DataFrame([{
            'hour': dt.hour,
            'day': dt.day,
            'weekday': dt.weekday(),
            'month': dt.month
        }])

        # Pad with dummy target (e.g., 0) and scale
        dummy_target = [0]
        scaled_input = scaler.transform(np.hstack([features.values, [dummy_target]]))[:, :-1]
        scaled_input = scaled_input.reshape((1, 1, len(FEATURES)))  # Reshape for LSTM

        # Predict
        scaled_output = model.predict(scaled_input)

        # Pad back for inverse transform
        padded_output = np.hstack([np.zeros((1, scaler.n_features_in_ - 1)), scaled_output])
        final_output = scaler.inverse_transform(padded_output)[0, -1]

        return jsonify({"predicted_power": round(float(final_output), 2)})

    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == '__main__':
    app.run(debug=True)
