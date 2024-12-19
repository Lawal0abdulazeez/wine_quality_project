# config.py

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data related configurations
DATA_CONFIG = {
    # Paths for data
    'raw_data_path': os.path.join(BASE_DIR, 'data', 'raw'),
    'processed_data_path': os.path.join(BASE_DIR, 'data', 'processed'),
    'models_path': os.path.join(BASE_DIR, 'models', 'saved_models'),
    
    # Data processing parameters
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'target_column': 'quality',
    
    # File names
    'red_wine_file': 'winequality-red.csv',
    'white_wine_file': 'winequality-white.csv'
}

# Model related configurations
MODEL_CONFIG = {
    # Model parameters
    'random_forest_params': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    
    'gradient_boosting_params': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    },
    
    'svm_params': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    }
}

# Feature related configurations
FEATURE_CONFIG = {
    # Features to use in the model
    'numeric_features': [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    ],
    
    'categorical_features': [
        'wine_type'
    ],
    
    # Feature engineering parameters
    'outlier_threshold': 3,  # Number of standard deviations for outlier detection
    'scaling_method': 'standard'  # or 'minmax', 'robust'
}

# Training configurations
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'early_stopping_patience': 10,
    'learning_rate': 0.001
}

# MLflow configurations
MLFLOW_CONFIG = {
    'tracking_uri': os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
    'experiment_name': 'wine_quality_prediction',
    'run_name_prefix': 'model_v'
}

# API configurations
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'model_version': 'v1',
    'api_version': 'v1'
}

# Monitoring configurations
MONITORING_CONFIG = {
    'metrics_path': os.path.join(BASE_DIR, 'metrics'),
    'logging_path': os.path.join(BASE_DIR, 'logs'),
    'model_drift_threshold': 0.1,
    'data_drift_threshold': 0.1
}

# Database configurations (if needed)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'wine_quality'),
    'username': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

def get_config():
    """
    Returns all configurations as a single dictionary.
    Useful when you need to pass all configs to a component.
    """
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'feature': FEATURE_CONFIG,
        'training': TRAINING_CONFIG,
        'mlflow': MLFLOW_CONFIG,
        'api': API_CONFIG,
        'monitoring': MONITORING_CONFIG,
        'db': DB_CONFIG
    }

def validate_paths():
    """
    Validates and creates necessary directories if they don't exist.
    """
    paths = [
        DATA_CONFIG['raw_data_path'],
        DATA_CONFIG['processed_data_path'],
        DATA_CONFIG['models_path'],
        MONITORING_CONFIG['metrics_path'],
        MONITORING_CONFIG['logging_path']
    ]
    
    for path in paths:
        os.makedirs(path, exist_ok=True)

# Initialize paths when config is imported
validate_paths()