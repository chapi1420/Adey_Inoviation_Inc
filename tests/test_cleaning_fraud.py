import unittest
import pandas as pd
import os
import numpy as np
import sys
sys.path.append(os.path.abspath("Adey_Inoviation_Inc/scripts"))

from scripts.data_merger import DataMerger  
from scripts.fraud_data_cleaning import FraudDataCleaner  

class TestDataMerger(unittest.TestCase):
    def setUp(self):
        """Set up temporary datasets for testing."""
        self.fraud_test_file = "test_fraud_data.csv"
        self.ip_test_file = "test_ip_data.csv"
        self.merged_test_file = "test_merged_data.csv"
        
        fraud_data = {
            "ip_address": ["192.168.1.1", "10.0.0.1", "172.16.0.1"],
            "amount": [100, 200, 300]
        }
        
        ip_data = {
            "lower_bound_ip_address": [int(ipaddress.ip_address("192.168.1.0")), int(ipaddress.ip_address("10.0.0.0"))],
            "upper_bound_ip_address": [int(ipaddress.ip_address("192.168.1.255")), int(ipaddress.ip_address("10.0.0.255"))],
            "country": ["USA", "Canada"]
        }
        
        pd.DataFrame(fraud_data).to_csv(self.fraud_test_file, index=False)
        pd.DataFrame(ip_data).to_csv(self.ip_test_file, index=False)
        
        self.merger = DataMerger(self.fraud_test_file, self.ip_test_file)
    
    def tearDown(self):
        """Remove test files after tests run."""
        for file in [self.fraud_test_file, self.ip_test_file, self.merged_test_file]:
            if os.path.exists(file):
                os.remove(file)
    
    def test_load_data(self):
        """Test loading data from CSV files."""
        self.merger.load_data()
        self.assertIsNotNone(self.merger.fraud_data)
        self.assertIsNotNone(self.merger.ip_data)
    
    def test_convert_ip_to_integer(self):
        """Test conversion of IP addresses to integers."""
        self.merger.load_data()
        self.merger.convert_ip_to_integer()
        self.assertTrue("ip_integer" in self.merger.fraud_data.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(self.merger.fraud_data["ip_integer"]))
    
    def test_merge_with_ip_data(self):
        """Test merging fraud data with IP data."""
        self.merger.load_data()
        self.merger.convert_ip_to_integer()
        self.merger.merge_with_ip_data()
        self.assertTrue("country" in self.merger.fraud_data.columns)
    
    def test_save_merged_data(self):
        """Test saving the merged dataset."""
        self.merger.load_data()
        self.merger.convert_ip_to_integer()
        self.merger.merge_with_ip_data()
        self.merger.save_merged_data(self.merged_test_file)
        self.assertTrue(os.path.exists(self.merged_test_file))

class TestFraudDataCleaner(unittest.TestCase):
    def setUp(self):
        """Set up temporary fraud dataset for testing."""
        self.fraud_test_file = "test_fraud_cleaner.csv"
        self.cleaned_test_file = "test_cleaned_fraud_data.csv"
        
        fraud_data = {
            "user_id": [1, 1, 2],
            "purchase_time": ["2023-01-01 12:00:00", "2023-01-01 12:00:00", "2023-01-02 14:30:00"],
            "device_id": ["abc123", "abc123", "xyz789"],
            "age": [25, None, 30],
            "purchase_value": [100, 200, None],
            "source": ["SEO", "SEO", None],
            "browser": ["Chrome", "Firefox", "Chrome"],
            "sex": ["M", None, "F"]
        }
        
        pd.DataFrame(fraud_data).to_csv(self.fraud_test_file, index=False)
        self.cleaner = FraudDataCleaner(self.fraud_test_file)
    
    def tearDown(self):
        """Remove test files after tests run."""
        for file in [self.fraud_test_file, self.cleaned_test_file]:
            if os.path.exists(file):
                os.remove(file)
    
    def test_load_data(self):
        """Test loading data."""
        self.cleaner.load_data()
        self.assertIsNotNone(self.cleaner.data)
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        self.cleaner.load_data()
        self.cleaner.remove_duplicates()
        self.assertEqual(len(self.cleaner.data), 2)
    
    def test_fix_data_types(self):
        """Test data type corrections."""
        self.cleaner.load_data()
        self.cleaner.fix_data_types()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.cleaner.data["purchase_time"]))
    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        self.cleaner.load_data()
        self.cleaner.handle_missing_values()
        self.assertFalse(self.cleaner.data.isnull().values.any())
    
    def test_detect_outliers(self):
        """Test outlier detection and removal."""
        self.cleaner.load_data()
        self.cleaner.detect_outliers()
        self.assertTrue("purchase_value" in self.cleaner.data.columns)
    
    def test_save_cleaned_data(self):
        """Test saving cleaned data."""
        self.cleaner.load_data()
        self.cleaner.remove_duplicates()
        self.cleaner.fix_data_types()
        self.cleaner.handle_missing_values()
        self.cleaner.detect_outliers()
        self.cleaner.save_cleaned_data(self.cleaned_test_file)
        self.assertTrue(os.path.exists(self.cleaned_test_file))

if __name__ == "__main__":
    unittest.main()
