import pandas as pd
from flask import Flask, jsonify

class FraudDetectionBackend:
    def __init__(self, data_path):
        """Initialize Flask app & load dataset."""
        self.app = Flask(__name__)
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path)
        self.setup_routes()

    def setup_routes(self):
        """Define API endpoints."""

        @self.app.route("/", methods=["GET"])
        def home():
            return jsonify({"message": "Fraud Detection Dashboard API is running!"})

        @self.app.route("/summary", methods=["GET"])
        def summary():
            """Returns summary statistics (total transactions, fraud count, fraud %)."""
            total_transactions = len(self.data)
            fraud_cases = self.data[self.data["class"] == 1].shape[0]
            fraud_percentage = (fraud_cases / total_transactions) * 100

            return jsonify({
                "total_transactions": total_transactions,
                "fraud_cases": fraud_cases,
                "fraud_percentage": round(fraud_percentage, 2)
            })

        @self.app.route("/fraud-trends", methods=["GET"])
        def fraud_trends():
            """Returns fraud counts per day."""
            self.data["purchase_time"] = pd.to_datetime(self.data["purchase_time"])
            daily_fraud = self.data[self.data["class"] == 1].groupby(self.data["purchase_time"].dt.date).size()

            return jsonify(daily_fraud.to_dict())

    def run(self):
        """Start Flask API."""
        self.app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    backend = FraudDetectionBackend(data_path="/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/featured/processed_fraud_data.csv")
    backend.run()
