import pandas as pd
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FraudDataCleaner:
    def __init__(self, file_path: str):
        """Initialize with dataset file path."""
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            logging.info(f"✅ Data loaded successfully from {self.file_path}")
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")

    def remove_duplicates(self):
        """Removes duplicate transactions based on key features."""
        if self.data is not None:
            initial_shape = self.data.shape
            self.data.drop_duplicates(subset=["user_id", "purchase_time", "device_id"], inplace=True)
            final_shape = self.data.shape
            logging.info(f"🔄 Removed {initial_shape[0] - final_shape[0]} duplicate transactions.")
        else:
            logging.error("❌ Data is not loaded. Call `load_data()` first.")

    def fix_data_types(self):
        """Converts columns to correct data types."""
        if self.data is not None:
            if "purchase_time" in self.data.columns:
                self.data["purchase_time"] = pd.to_datetime(self.data["purchase_time"], errors="coerce")
                logging.info("📅 Converted 'purchase_time' to datetime.")

            if "signup_time" in self.data.columns:
                self.data["signup_time"] = pd.to_datetime(self.data["signup_time"], errors="coerce")
                logging.info("📅 Converted 'signup_time' to datetime.")

            # Convert numeric columns
            numeric_cols = ["age", "purchase_value"]
            for col in numeric_cols:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                    logging.info(f"🔢 Converted '{col}' to numeric type.")

        else:
            logging.error("❌ Data is not loaded.")

    def handle_missing_values(self):
        """Handles missing values based on column type."""
        if self.data is not None:
            # Fill missing ages with median
            if "age" in self.data.columns:
                median_age = self.data["age"].median()
                self.data["age"].fillna(median_age, inplace=True)
                logging.info(f"🧪 Filled missing 'age' values with median: {median_age}")

            # Fill missing purchase_value with median
            if "purchase_value" in self.data.columns:
                median_value = self.data["purchase_value"].median()
                self.data["purchase_value"].fillna(median_value, inplace=True)
                logging.info(f"🧪 Filled missing 'purchase_value' values with median: {median_value}")

            # Fill missing categorical values with most common category
            categorical_cols = ["source", "browser", "sex"]
            for col in categorical_cols:
                if col in self.data.columns:
                    most_frequent = self.data[col].mode()[0]
                    self.data[col].fillna(most_frequent, inplace=True)
                    logging.info(f"🔠 Filled missing '{col}' with most frequent: {most_frequent}")

        else:
            logging.error("❌ Data is not loaded.")

    def detect_outliers(self, threshold=3):
        """Detects and removes outliers based on Z-score method for purchase value."""
        if self.data is not None:
            if "purchase_value" in self.data.columns:
                mean_val = self.data["purchase_value"].mean()
                std_val = self.data["purchase_value"].std()
                z_scores = (self.data["purchase_value"] - mean_val) / std_val

                outliers = self.data[np.abs(z_scores) > threshold]
                self.data = self.data[np.abs(z_scores) <= threshold]

                logging.info(f"🚨 Removed {len(outliers)} outliers based on Z-score threshold {threshold}.")
        else:
            logging.error("❌ Data is not loaded.")

    def save_cleaned_data(self, output_path: str):
        """Saves the cleaned dataset to a new CSV file."""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            logging.info(f"✅ Cleaned data saved to {output_path}")
        else:
            logging.error("❌ No data available to save.")

# Example Usage:
if __name__ == "__main__":
    fraud_cleaner = FraudDataCleaner("/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/missingHandled/Fraud_Data.csv")
    
    # Step 1: Load data
    fraud_cleaner.load_data()

    # Step 2: Remove duplicates
    fraud_cleaner.remove_duplicates()

    # Step 3: Fix incorrect data types
    fraud_cleaner.fix_data_types()

    # Step 4: Handle missing values
    fraud_cleaner.handle_missing_values()

    # Step 5: Detect and remove outliers
    fraud_cleaner.detect_outliers(threshold=3)

    # Step 6: Save cleaned data
    fraud_cleaner.save_cleaned_data("/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_Fraud_Data.csv")
