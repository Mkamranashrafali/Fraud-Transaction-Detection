from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- DATA LOADING AND PREDICTION CLASS ---
class PickleData:
    def load_object(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found")
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def predict_with_loaded_model(self, model_path, scaler_path, sample_data):
        model = self.load_object(model_path)
        scaler = self.load_object(scaler_path)
        sample_scaled = scaler.transform(sample_data)
        prediction = model.predict(sample_scaled)
        return prediction

# --- INITIALIZATION ---
pickler = PickleData()
feature_cols = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 'ChannelEncoded', 'LoginAttempts']
channel_mapping = {"Online": 0, "Branch": 1, "ATM": 2}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api')
def api_info():
    return jsonify({
        "message": "Fraud Detection API",
        "endpoints": {
            "/health": "GET - Check API health",
            "/predict": "POST - Predict fraud for a transaction"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Validate required fields
        required_fields = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 'Channel', 'LoginAttempts']
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
        
        # Encode channel
        channel = data['Channel']
        if channel not in channel_mapping:
            return jsonify({"error": f"Invalid channel. Must be one of: {list(channel_mapping.keys())}"}), 400
        
        # Prepare input data
        input_data = {
            'TransactionAmount': [data['TransactionAmount']],
            'CustomerAge': [data['CustomerAge']],
            'AccountBalance': [data['AccountBalance']],
            'ChannelEncoded': [channel_mapping[channel]],
            'LoginAttempts': [data['LoginAttempts']]
        }
        input_df = pd.DataFrame(input_data)[feature_cols]
        
        # Make prediction
        prediction = pickler.predict_with_loaded_model(
            model_path="svm_model.pkl",
            scaler_path="scaler.pkl",
            sample_data=input_df
        )
        
        result = {
            "prediction": int(prediction[0]),
            "is_fraud": bool(prediction[0] == 1),
            "message": "High Risk: Fraud Detected!" if prediction[0] == 1 else "Low Risk: Transaction is Legitimate"
        }
        
        return jsonify(result)
    
    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf)}), 500
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
