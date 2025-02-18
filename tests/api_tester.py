import unittest
import requests

class TestFraudDetectionAPI(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:5000"  

    def test_home_endpoint(self):
        """Test if the home route is working."""
        response = requests.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Fraud Detection API", response.json()["message"])

    def test_predict_endpoint_valid(self):
        """Test fraud prediction with valid input."""
        test_data = {
            "purchase_value": 120.5,
            "age": 30,
            "device_id": 12345,
            "source": 2,  
            "browser": 1, 
            "ip_address": 3232235777  
        }

        response = requests.post(f"{self.BASE_URL}/predict", json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("fraud_prediction", response.json())

    def test_predict_endpoint_invalid(self):
        """Test fraud prediction with missing or invalid input."""
        test_data = {
            "purchase_value": "invalid",  
        }

        response = requests.post(f"{self.BASE_URL}/predict", json=test_data)
        self.assertEqual(response.status_code, 200)  
        self.assertIn("error", response.json())

if __name__ == "__main__":
    unittest.main()
