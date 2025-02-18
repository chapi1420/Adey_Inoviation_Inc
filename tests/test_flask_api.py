import unittest
import requests

class TestFraudDetectionAPI(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:5000"  # Flask API URL

    def test_home_endpoint(self):
        """Test if the home route is working."""
        response = requests.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Fraud Detection Dashboard API", response.json()["message"])

    def test_summary_endpoint(self):
        """Test the summary statistics endpoint."""
        response = requests.get(f"{self.BASE_URL}/summary")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("total_transactions", data)
        self.assertIn("fraud_cases", data)
        self.assertIn("fraud_percentage", data)

    def test_fraud_trends_endpoint(self):
        """Test the fraud trends endpoint."""
        response = requests.get(f"{self.BASE_URL}/fraud-trends")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIsInstance(data, dict)  

if __name__ == "__main__":
    unittest.main()
