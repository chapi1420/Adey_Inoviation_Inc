import pandas as pd
import logging
import ipaddress

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataMerger:
    def __init__(self, fraud_data_path: str, ip_data_path: str):
        """Initialize with paths for fraud transaction and IP address datasets."""
        self.fraud_data_path = fraud_data_path
        self.ip_data_path = ip_data_path
        self.fraud_data = None
        self.ip_data = None

    def load_data(self):
        """Loads both datasets."""
        try:
            self.fraud_data = pd.read_csv(self.fraud_data_path)
            self.ip_data = pd.read_csv(self.ip_data_path)
            logging.info("✅ Successfully loaded both datasets.")
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")

    def convert_ip_to_integer(self):
        """Converts IP addresses to integer format in the fraud data."""
        if self.fraud_data is not None:
            try:
                self.fraud_data["ip_integer"] = self.fraud_data["ip_address"].apply(
                    lambda ip: int(ipaddress.ip_address(ip)) if isinstance(ip, str) else ip
                )
                logging.info("🔢 Converted IP addresses to integer format.")
            except Exception as e:
                logging.error(f"❌ Error converting IP addresses: {e}")
        else:
            logging.error("❌ Fraud data is not loaded.")

    def merge_with_ip_data(self):
        """Merges fraud data with IP-country mapping based on IP range."""
        if self.fraud_data is not None and self.ip_data is not None:
            self.ip_data["lower_bound_ip_address"] = self.ip_data["lower_bound_ip_address"].astype(int)
            self.ip_data["upper_bound_ip_address"] = self.ip_data["upper_bound_ip_address"].astype(int)

            # Merge using conditions (IP must be between lower and upper bound)
            merged_data = self.fraud_data.merge(self.ip_data, how="left",
                left_on="ip_integer",
                right_on="lower_bound_ip_address"
            )

            # Keep only necessary columns
            merged_data.drop(columns=["lower_bound_ip_address", "upper_bound_ip_address"], inplace=True)
            self.fraud_data = merged_data
            logging.info("🔗 Merged fraud data with country info based on IP addresses.")
        else:
            logging.error("❌ One or both datasets are missing.")

    def save_merged_data(self, output_path: str):
        """Saves the merged dataset to a new CSV file."""
        if self.fraud_data is not None:
            self.fraud_data.to_csv(output_path, index=False)
            logging.info(f"✅ Merged dataset saved to {output_path}")
        else:
            logging.error("❌ No data available to save.")

# Example Usage:
if __name__ == "__main__":
    merger = DataMerger("/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_Fraud_Data.csv", "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_IpAddress_to_Country.csv")

    # Load datasets
    merger.load_data()

    # Convert IP addresses
    merger.convert_ip_to_integer()

    # Merge with IP country data
    merger.merge_with_ip_data()

    # Save merged dataset
    merger.save_merged_data("/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/Fraud_And_Ip_merged.csv")
