import os
import pandas as pd
import numpy as np
import mlflow
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Input, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from pathlib import Path

# Force TensorFlow to use CPU (to avoid CUDA errors)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class FraudDeepModelTrainer:
    def __init__(self, fraud_data_path: str, credit_data_path: str, output_path: str, model_type: str = 'mlp'):
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
        """Separate features and target, standardize numerical data, then split into train and test sets."""
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Standardize numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def select_model(self, input_shape, is_sequence=False):
        """Initialize the deep learning model based on user selection."""
        if self.model_type == 'mlp':
            self.model = Sequential([
                Input(shape=(input_shape,)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
        elif self.model_type == 'cnn':
            self.model = Sequential([
                Input(shape=(input_shape, 1)),
                Conv1D(32, kernel_size=3, activation='relu'),
                Flatten(),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
        elif self.model_type == 'lstm':
            self.model = Sequential([
                Input(shape=(input_shape, 1)),
                LSTM(32),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
        else:
            raise ValueError("Invalid model type. Choose 'mlp', 'cnn', or 'lstm'.")
        
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train the selected deep learning model and evaluate it."""
        # Handle class imbalance
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Reshape data for CNN & LSTM
        if len(X_train.shape) == 2 and self.model_type in ['cnn', 'lstm']:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test),
                       verbose=1, class_weight=class_weight_dict)
        y_pred = (self.model.predict(X_test) > 0.5).astype('int32')
        report = classification_report(y_test, y_pred, zero_division=1)
        print(report)
        return report
    
    def log_experiment(self, report):
        """Log model training details using MLflow."""
        mlflow.set_experiment("fraud_detection")
        with mlflow.start_run():
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_text(report, "classification_report.txt")
            mlflow.keras.log_model(self.model, "fraud_model")
    
    def save_model(self):
        """Save the trained model in MLflow and Pickle format."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.output_path)
        print(f"Model saved successfully to {self.output_path} (Keras format)")
        
        # Save as Pickle
        pickle_path = self.output_path + ".pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model also saved as {pickle_path} (Pickle format)")
    
    def run_pipeline(self):
        print("Loading data...")
        self.load_data()
        
        print("Preprocessing fraud dataset...")
        X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = self.preprocess_data(self.fraud_df, 'class')
        
        print("Preprocessing credit dataset...")
        X_train_credit, X_test_credit, y_train_credit, y_test_credit = self.preprocess_data(self.credit_df, 'Class')
        
        print(f"Selecting {self.model_type} model...")
        self.select_model(X_train_fraud.shape[1])
        
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
    OUTPUT_PATH = "/home/nahomnadew/Desktop/10x/week8/Adey_Inoviation_Inc/Models/fraud_deep_model"
    MODEL_TYPE = 'lstm'  # Options: 'mlp', 'cnn', 'lstm'
    
    trainer = FraudDeepModelTrainer(FRAUD_DATA_PATH, CREDIT_DATA_PATH, OUTPUT_PATH, MODEL_TYPE)
    trainer.run_pipeline()
