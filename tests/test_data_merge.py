import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath("Adey_Inoviation_Inc/scripts"))
from   scripts.data_merger import DataMerger  
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

if __name__ == "__main__":
    unittest.main()
