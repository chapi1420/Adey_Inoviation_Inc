from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
import pandas as pd

class DatasetPreparer:
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = RobustScaler()
        
    def prepare_fraud_data(self, df):
        """Prepare fraud dataset with temporal splitting"""
        # Time-based split
        df = df.sort_values('purchase_time')
        split_idx = int(len(df) * (1 - self.test_size))
        
        X_train = df.iloc[:split_idx].drop(columns=['class'])
        y_train = df.iloc[:split_idx]['class']
        X_test = df.iloc[split_idx:].drop(columns=['class'])
        y_test = df.iloc[split_idx:]['class']
        
        # Handle class imbalance
        smote = SMOTE(sampling_strategy=0.3, random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        
        # Scale numerical features
        num_cols = ['purchase_value', 'signup_to_purchase_hours', 'age']
        X_res[num_cols] = self.scaler.fit_transform(X_res[num_cols])
        X_test[num_cols] = self.scaler.transform(X_test[num_cols])
        
        return X_res, X_test, y_res, y_test
    
    def prepare_credit_data(self, df):
        """Prepare credit card dataset"""
        # Random split due to anonymized time
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        
        # Scale using RobustScaler (handles outliers)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_sets(self):
        """Final feature sets after preparation"""
        return {
            'fraud': ['purchase_value', 'cross_border', 'time_since_last_purchase',
                     'device_usage_freq', 'purchase_hour', 'signup_to_purchase_hours'],
            'credit': ['V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'Amount_scaled']
        }

# Usage Example
if __name__ == "__main__":
    # Load processed data
    fraud_data = pd.read_csv('/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_Fraud_Data.csv')
    credit_data = pd.read_csv('/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_creditcard.csv')
    
    # Prepare datasets
    preparer = DatasetPreparer()
    
    # Fraud data preparation
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = preparer.prepare_fraud_data(fraud_data)
    
    # Credit data preparation
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = preparer.prepare_credit_data(credit_data)