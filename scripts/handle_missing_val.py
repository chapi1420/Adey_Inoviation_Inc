import pandas as pd
from typing import Optional, Union

class DataPreprocessor:
    def __init__(self, file_path: str):
        """Initialize the class with a dataset file path."""
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
        self.numerical_cols = []
        self.categorical_cols = []

    def load_data(self):
        """Loads data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            self._identify_column_types()
            print(f"✅ Data loaded successfully from {self.file_path}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def _identify_column_types(self):
        """Identifies numerical and categorical columns."""
        if self.data is not None:
            self.numerical_cols = self.data.select_dtypes(include=["number"]).columns.tolist()
            self.categorical_cols = self.data.select_dtypes(include=["object", "category"]).columns.tolist()
            print(f"🔢 Numerical Columns: {self.numerical_cols}")
            print(f"🔤 Categorical Columns: {self.categorical_cols}")

    def check_missing_values(self):
        """Returns the count of missing values per column."""
        if self.data is not None:
            missing_values = self.data.isnull().sum()
            print("📊 Missing values per column:\n", missing_values[missing_values > 0])
            return missing_values
        else:
            print("❌ Data is not loaded. Call `load_data()` first.")

    def handle_missing_values(self, method: str = "auto", fill_values: Optional[dict] = None):
        """
        Handles missing values in the dataset.

        Parameters:
        - method: str
          - "auto" (default): Uses mean/median for numerical, mode for categorical.
          - "drop": Removes rows with missing values.
          - "fill": Uses specified fill values (or defaults if not provided).
        - fill_values: dict, Custom fill values for specific columns.
        """
        if self.data is None:
            print("❌ Data is not loaded. Call `load_data()` first.")
            return

        if method == "drop":
            self.data.dropna(inplace=True)
            print("🗑️ Dropped rows with missing values.")
        elif method == "fill" or method == "auto":
            for col in self.data.columns:
                if self.data[col].isnull().sum() > 0:  # Only process columns with missing values
                    if fill_values and col in fill_values:
                        self.data[col].fillna(fill_values[col], inplace=True)
                        print(f"🧪 Filled {col} with custom value: {fill_values[col]}")
                    elif col in self.numerical_cols:
                        fill_val = self.data[col].mean() if method == "auto" else self.data[col].median()
                        self.data[col].fillna(fill_val, inplace=True)
                        print(f"📈 Filled {col} with {'mean' if method == 'auto' else 'median'}: {fill_val}")
                    elif col in self.categorical_cols:
                        fill_val = self.data[col].mode()[0]
                        self.data[col].fillna(fill_val, inplace=True)
                        print(f"🔠 Filled {col} with mode: {fill_val}")
        else:
            print("❌ Invalid method! Use 'drop', 'fill', or 'auto'.")

    def save_cleaned_data(self, output_path: str):
        """Saves the cleaned dataset to a new CSV file."""
        if self.data is not None:
            self.data.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to {output_path}")
        else:
            print("❌ No data available to save.")

# Example Usage:
if __name__ == "__main__":
    preprocessor = DataPreprocessor("Fraud_Data.csv")
    preprocessor.load_data()
    preprocessor.check_missing_values()
    
    # Auto-fill: Uses mean for numerical, mode for categorical
    preprocessor.handle_missing_values(method="auto")

    # OR, specify custom fill values
    # preprocessor.handle_missing_values(method="fill", fill_values={"age": 30, "browser": "Chrome"})

    preprocessor.save_cleaned_data("Fraud_Data_Cleaned.csv")
