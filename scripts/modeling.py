import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from pathlib import Path

class FraudModelTrainer:
    def __init__(self, fraud_data_path: str, credit_data_path: str, output_path: str, model_type: str = 'random_forest'):
        self.fraud_data_path = fraud_data_path
        self.credit_data_path = credit_data_path
        self.output_path = output_path
        self.model_type = model_type.lower()
        self.fraud_df = None
        self.credit_df = None
        self.model = None
    
    def load_data(self):
        """Load and preprocess fraud and credit card datasets."""
        self.fraud_df = pd.read_csv(self.fraud_data_path)
        self.credit_df = pd.read_csv(self.credit_data_path)
    
    def preprocess_data(self, df: pd.DataFrame, target_col: str):
        """Separate features and target, then split into train and test sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def select_model(self):
        """Initialize the model based on user selection."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression()
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier()
        else:
            raise ValueError("Invalid model type. Choose 'random_forest', 'logistic_regression', or 'decision_tree'.")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train the selected model and evaluate it."""
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)
        return report
    
    def log_experiment(self, report):
        """Log model training details using MLflow."""
        mlflow.set_experiment("fraud_detection")
        with mlflow.start_run():
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_text(report, "classification_report.txt")
            mlflow.sklearn.log_model(self.model, "fraud_model")
    
    def save_model(self):
        """Save the trained model."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(self.model, self.output_path)
        print(f"Model saved successfully to {self.output_path}")
    
    def run_pipeline(self):
        print("Loading data...")
        self.load_data()
        
        print("Preprocessing fraud dataset...")
        X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = self.preprocess_data(self.fraud_df, 'class')
        
        print("Preprocessing credit dataset...")
        X_train_credit, X_test_credit, y_train_credit, y_test_credit = self.preprocess_data(self.credit_df, 'Class')
        
        print(f"Selecting {self.model_type} model...")
        self.select_model()
        
        print("Training and evaluating on fraud data...")
        report_fraud = self.train_and_evaluate(X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud)
        
        print("Training and evaluating on credit card data...")
        report_credit = self.train_and_evaluate(X_train_credit, X_test_credit, y_train_credit, y_test_credit)
        
        print("Logging experiment...")
        self.log_experiment(report_fraud)
        self.log_experiment(report_credit)
        
        print("Saving trained model...")
        self.save_model()
        print("Pipeline complete!")

if __name__ == "__main__":
    FRAUD_DATA_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/featured/processed_fraud_data.csv"
    CREDIT_DATA_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Data/cleaned/cleaned_creditcard.csv"
    OUTPUT_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Models/decision_tree_fraud_model"
    MODEL_TYPE = 'decision_tree'  # Options: 'random_forest', 'logistic_regression', 'decision_tree'
    
    trainer = FraudModelTrainer(FRAUD_DATA_PATH, CREDIT_DATA_PATH, OUTPUT_PATH, MODEL_TYPE)
    trainer.run_pipeline()
