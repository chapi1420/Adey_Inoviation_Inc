import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath("Adey_Inoviation_Inc/scripts"))
from   scripts.data_cleaning import * 
class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        """Set up a temporary dataset for testing."""
        self.test_file = "test_fraud_data.csv"
        self.cleaned_file = "test_fraud_data_cleaned.csv"
        
        data = {
            "purchase_time": ["2023-01-01 10:00:00", "2023-01-01 10:00:00", "2023-02-01 15:00:00"],
            "amount": ["100", "100", "200"],
            "category": ["A", "A", "B"]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(self.test_file, index=False)
        self.cleaner = DataCleaner(self.test_file)
    
    def tearDown(self):
        """Remove test files after tests run."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.cleaned_file):
            os.remove(self.cleaned_file)

    def test_load_data(self):
        """Test loading data from CSV."""
        self.cleaner.load_data()
        self.assertIsNotNone(self.cleaner.data)
        self.assertEqual(self.cleaner.data.shape, (3, 3))
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        self.cleaner.load_data()
        self.cleaner.remove_duplicates()
        self.assertEqual(self.cleaner.data.shape, (2, 3))
    
    def test_correct_data_types(self):
        """Test data type correction."""
        self.cleaner.load_data()
        self.cleaner.correct_data_types()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.cleaner.data["purchase_time"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.cleaner.data["amount"]))
        self.assertTrue(pd.api.types.is_categorical_dtype(self.cleaner.data["category"]))
    
    def test_save_cleaned_data(self):
        """Test saving cleaned data."""
        self.cleaner.load_data()
        self.cleaner.remove_duplicates()
        self.cleaner.correct_data_types()
        self.cleaner.save_cleaned_data(self.cleaned_file)
        self.assertTrue(os.path.exists(self.cleaned_file))

if __name__ == "__main__":
    unittest.main()