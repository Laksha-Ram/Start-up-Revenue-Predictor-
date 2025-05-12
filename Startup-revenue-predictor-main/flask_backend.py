from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import xgboost as xgb

app = Flask(__name__)

# Load LSTM model
lstm_model = tf.keras.models.load_model("lstm_funding_model.keras")

# Load XGBoost model from JSON format
xgb_model = xgb.Booster()
xgb_model.load_model("xgboost_startup_model_v3.json")

@app.route('/')
def home():
    return "Flask API for ML & DL models is running!"

@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    try:
        data = request.get_json()
        input_data = np.array(data['input_data'])

        # Validate shape: should be (1, 12, 6)
        if input_data.shape != (12, 6):
            return jsonify({'error': 'Input data must have shape (12, 6)'}), 400

        input_data = input_data.reshape(1, 12, 6)  # Reshape for LSTM
        prediction = lstm_model.predict(input_data).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict_xgboost', methods=['POST'])
def predict_xgboost():
    try:
        data = request.get_json()
        input_data = np.array(data['input_data']).reshape(1, -1)
        dmatrix = xgb.DMatrix(input_data)  # Convert input to DMatrix
        prediction = xgb_model.predict(dmatrix).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
