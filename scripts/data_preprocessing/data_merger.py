import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

class FraudDataProcessor:
    def __init__(self, input_path: str, ip_mapping_path: str, output_path: str, scaling_method: str = 'standard', encoding_method: str = 'label'):
        self.input_path = input_path
        self.ip_mapping_path = ip_mapping_path
        self.output_path = output_path
        self.scaling_method = scaling_method.lower()
        self.encoding_method = encoding_method.lower()
        self.df = None
        self.ip_mapping = None
        self.scaler = None
        self.label_encoders = {}
    
    def load_data(self):
        """Load processed fraud dataset and IP mapping dataset."""
        self.df = pd.read_csv(self.input_path)
        self.ip_mapping = pd.read_csv(self.ip_mapping_path)
    
    def map_ip_to_country(self):
        """Map IP addresses to corresponding countries."""
        self.df['ip_address_numeric'] = self.df['ip_address'].apply(self.ip_to_numeric)
        
        def find_country(ip):
            match = self.ip_mapping[(self.ip_mapping['lower_bound_ip_address'] <= ip) &
                                    (self.ip_mapping['upper_bound_ip_address'] >= ip)]
            return match['country'].values[0] if not match.empty else 'Unknown'
        
        self.df['ip_country'] = self.df['ip_address_numeric'].apply(find_country)
        self.df.drop(columns=['ip_address', 'ip_address_numeric'], inplace=True)
    
    def ip_to_numeric(self, ip_str):
        """Convert IP address from string to numeric format."""
        parts = list(map(int, ip_str.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    
    def encode_categorical_features(self):
        """Convert all non-numeric categorical features into numeric format."""
        categorical_cols = ['device_id', 'source', 'browser', 'sex', 'ip_country']
        for col in categorical_cols:
            if col in self.df.columns:
                self.label_encoders[col] = LabelEncoder()
                self.df[col + '_encoded'] = self.label_encoders[col].fit_transform(self.df[col])
                self.df.drop(columns=[col], inplace=True)
    
    def select_features(self):
        """Select numerical features for scaling."""
        self.numeric_cols = ['purchase_value', 'age', 'time_since_signup',
                             'signup_hour', 'signup_dayofweek', 'purchase_hour', 'purchase_dayofweek',
                             'device_id_encoded', 'source_encoded', 'browser_encoded', 'sex_encoded', 'ip_country_encoded']
        self.df_numeric = self.df[self.numeric_cols]
    
    def apply_scaling(self):
        """Apply normalization or standardization."""
        if self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        self.df_scaled = self.df.copy()
        self.df_scaled[self.numeric_cols] = self.scaler.fit_transform(self.df_numeric)
    
    def save_processed_data(self):
        """Save processed dataset."""
        self.df_scaled.to_csv(self.output_path, index=False)
        print(f"Processed data saved to {self.output_path}")
    
    def run_pipeline(self):
        print("Loading data...")
        self.load_data()
        print("Mapping IP addresses to countries...")
        self.map_ip_to_country()
        print("Encoding categorical features...")
        self.encode_categorical_features()
        print("Selecting features for scaling...")
        self.select_features()
        print(f"Applying {self.scaling_method} scaling...")
        self.apply_scaling()
        print("Saving processed data...")
        self.save_processed_data()
        print("Processing complete!")

if __name__ == "__main__":
    INPUT_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/featured/feature_engineered_fraud.csv"
    IP_MAPPING_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/mappings/ip_address_to_country.csv"
    OUTPUT_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/featured/processed_fraud_data.csv"
    SCALING_METHOD = 'standard'  # Change to 'minmax' for Min-Max scaling
    ENCODING_METHOD = 'label'  # Change to 'frequency' or 'target' for different encoding methods
    
    processor = FraudDataProcessor(INPUT_PATH, IP_MAPPING_PATH, OUTPUT_PATH, SCALING_METHOD, ENCODING_METHOD)
    processor.run_pipeline()
