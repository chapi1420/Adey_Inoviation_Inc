import pandas as pd
import logging
import numpy as np
import ipaddress

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IPAddressDataCleaner:
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
        """Removes duplicate rows from the dataset."""
        if self.data is not None:
            initial_shape = self.data.shape
            self.data.drop_duplicates(inplace=True)
            final_shape = self.data.shape
            logging.info(f"🔄 Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
        else:
            logging.error("❌ Data is not loaded. Call `load_data()` first.")

    def convert_ip_to_integer(self):
        """Converts IP addresses to integer format for efficient processing."""
        if self.data is not None:
            try:
                self.data["lower_bound_ip_address"] = self.data["lower_bound_ip_address"].apply(
                    lambda ip: int(ipaddress.ip_address(ip)) if isinstance(ip, str) else ip
                )
                self.data["upper_bound_ip_address"] = self.data["upper_bound_ip_address"].apply(
                    lambda ip: int(ipaddress.ip_address(ip)) if isinstance(ip, str) else ip
                )
                logging.info("🔢 Converted IP addresses to integer format.")
            except Exception as e:
                logging.error(f"❌ Error converting IP addresses: {e}")
        else:
            logging.error("❌ Data is not loaded.")

    def standardize_country_names(self):
        """Standardizes country names (removes extra spaces, capitalizes)."""
        if self.data is not None:
            self.data["country"] = self.data["country"].str.strip().str.title()
            logging.info("🌍 Standardized country names.")
        else:
            logging.error("❌ Data is not loaded.")

    def handle_missing_values(self):
        """Handles missing values by filling with 'Unknown' for country and interpolating IPs."""
        if self.data is not None:
            # Fill missing countries with "Unknown"
            self.data["country"].fillna("Unknown", inplace=True)
            
            # Fill missing IP values using interpolation (linear fill)
            self.data["lower_bound_ip_address"].interpolate(method="linear", inplace=True)
            self.data["upper_bound_ip_address"].interpolate(method="linear", inplace=True)

            logging.info("🧪 Handled missing values (filled missing countries, interpolated IPs).")
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
    ip_cleaner = IPAddressDataCleaner("/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/missingHandled/IpAddress_to_Country.csv")
    
    # Load data
    ip_cleaner.load_data()

    # Remove duplicates
    ip_cleaner.remove_duplicates()

    # Convert IP addresses to integers
    ip_cleaner.convert_ip_to_integer()

    # Standardize country names
    ip_cleaner.standardize_country_names()

    # Handle missing values
    ip_cleaner.handle_missing_values()

    # Save cleaned data
    ip_cleaner.save_cleaned_data("/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_IpAddress_to_Country.csv")
