import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

class FraudDetectionAPI:
    def __init__(self, model_path):
        """
        Initialize the Fraud Detection API.
        :param model_path: Path to the trained fraud detection model (.pkl file).
        """
        self.model_path = model_path
        self.model = self.load_model()
        self.app = Flask(__name__)
        self.setup_routes()

    def load_model(self):
        """Loads the trained fraud detection model from file."""
        with open(self.model_path, "rb") as file:
            model = pickle.load(file)
        print("✅ Model loaded successfully!")
        return model

    def setup_routes(self):
        """Defines the API endpoints."""
        @self.app.route("/", methods=["GET"])
        def home():
            return jsonify({"message": "Fraud Detection API is running!"})

        @self.app.route("/predict", methods=["POST"])
        def predict():
            """Endpoint to predict fraud based on user input."""
            try:
                # Get JSON data from request
                data = request.get_json()
                df = pd.DataFrame([data])  # Convert input to DataFrame
                
                # Ensure all values are numeric
                df = df.astype(float)
                
                # Predict fraud (0 = not fraud, 1 = fraud)
                prediction = self.model.predict(df)[0]
                probability = self.model.predict_proba(df)[0][1]

                return jsonify({
                    "fraud_prediction": int(prediction),
                    "fraud_probability": float(probability)
                })

            except Exception as e:
                return jsonify({"error": str(e)})

    def run(self):
        """Starts the Flask API server."""
        self.app.run(host="0.0.0.0", port=5000, debug=True)

# ---------------- USAGE ---------------- #
if __name__ == "__main__":
    api = FraudDetectionAPI(model_path="/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Models/fraud_model/model.pkl")
    api.run()
