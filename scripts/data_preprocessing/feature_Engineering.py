import pandas as pd
import numpy as np
from pathlib import Path

class FraudFeatureEngineer:
    def __init__(self, fraud_path: str, output_path: str):
        self.fraud_path = fraud_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        """Load fraud dataset and validate required columns."""
        self.df = pd.read_csv(self.fraud_path, parse_dates=['signup_time', 'purchase_time'])
        required_cols = {'user_id', 'signup_time', 'purchase_time', 'purchase_value', 'device_id',
                         'source', 'browser', 'sex', 'age', 'ip_address', 'class'}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            raise ValueError(f"Missing columns in fraud data: {missing}")

    def engineer_time_features(self):
        """Convert time-based features into ML-friendly formats."""
        self.df = self.df.copy()
        
        # Convert timestamps to UNIX time
        self.df['signup_timestamp'] = self.df['signup_time'].astype('int64') // 10**9
        self.df['purchase_timestamp'] = self.df['purchase_time'].astype('int64') // 10**9
        
        # Time difference features
        self.df['time_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds()
        
        # Extract temporal components
        self.df['signup_hour'] = self.df['signup_time'].dt.hour
        self.df['signup_dayofweek'] = self.df['signup_time'].dt.weekday
        self.df['purchase_hour'] = self.df['purchase_time'].dt.hour
        self.df['purchase_dayofweek'] = self.df['purchase_time'].dt.weekday
        self.df = self.df.drop(columns=['signup_time', 'purchase_time'])
        # Identify suspicious behaviors
        self.df['is_night_purchase'] = ((self.df['purchase_hour'] >= 22) | (self.df['purchase_hour'] <= 5)).astype(int)
        self.df['fast_purchase'] = (self.df['time_since_signup'] < 300).astype(int)  # Less than 5 minutes
        
        return self.df
    
    def save_processed_data(self):
        """Save processed data to CSV."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        print(f"Data saved successfully to {self.output_path}")
    
    def run_pipeline(self):
        print("Loading data...")
        self.load_data()
        print("Engineering time-based features...")
        self.engineer_time_features()
        print("Saving results...")
        self.save_processed_data()
        print("Feature engineering complete!")

if __name__ == "__main__":
    FRAUD_DATA_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/Fraud_And_Ip_merged.csv"
    OUTPUT_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/featured/feature_engineered_fraud.csv"
    
    fraud_engineer = FraudFeatureEngineer(FRAUD_DATA_PATH, OUTPUT_PATH)
    fraud_engineer.run_pipeline()
