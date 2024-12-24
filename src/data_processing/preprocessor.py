# src/data_processing/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and combine red and white wine datasets"""
        red_wine = pd.read_csv(f"{self.config['raw_data_path']}/winequality-red.csv", sep=';')
        white_wine = pd.read_csv(f"{self.config['raw_data_path']}/winequality-white.csv", sep=';')
        
        # Add wine type feature
        red_wine['wine_type'] = 'red'
        white_wine['wine_type'] = 'white'
        
        return pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Check for missing values
        missing_values = df.isnull().sum()
        
        # For numerical columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
    
    def handle_outliers(self, df, columns, n_std=3):
        """Handle outliers using the Z-score method"""
        for column in columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df[column] = df[column].mask(z_scores > n_std, df[column].median())
        return df
    
    def balance_dataset(self, X, y):
        """Balance the dataset using SMOTE"""
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=self.config['random_state'])
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def preprocess(self, df):
        """Main preprocessing pipeline"""
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Handle outliers for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = self.handle_outliers(df, numeric_columns)
        
        # Convert categorical variables
        df = pd.get_dummies(df, columns=['wine_type'])
        
        # Split features and target
        X = df.drop(self.config['target_column'], axis=1)
        y = df[self.config['target_column']]
        
        # Balance dataset
        X, y = self.balance_dataset(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Fit the scaler and transform the training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform the test data (scaler is already fitted on X_train)
        X_test_scaled = self.scaler.transform(X_test)
        

        # Define the path to save the preprocessor
        save_dir = os.path.join('models', 'saved_models')
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        preprocessor_path = os.path.join(save_dir, 'preprocessor.pkl')

        # Create a simple preprocessor (example: StandardScaler)
        preprocessor = self.scaler

        # Save the preprocessor
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to: {preprocessor_path}")

        return X_train_scaled, X_test_scaled, y_train, y_test