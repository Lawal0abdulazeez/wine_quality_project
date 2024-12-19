# src/validation/validate_model.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
from src.config import get_config

class ModelValidator:
    def __init__(self, config):
        self.config = config
        self.performance_threshold = 0.8
        
    def validate_model_performance(self, model, X_test, y_test):
        """Validate model performance meets minimum threshold"""
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        
        validation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        return all(score >= self.performance_threshold for score in validation_results.values())

# src/monitoring/model_monitor.py
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

class ModelMonitor:
    def __init__(self, config):
        self.config = config
        self.drift_threshold = config['monitoring']['model_drift_threshold']
        
    def calculate_data_drift(self, reference_data, current_data):
        """Calculate data drift using Evidently"""
        data_drift_profile = Profile(sections=[DataDriftProfileSection()])
        data_drift_profile.calculate(reference_data, current_data)
        return data_drift_profile.json()
    
    def monitor_model_performance(self, model, X, y_true):
        """Monitor model performance metrics"""
        y_pred = model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name='model_monitoring'):
            mlflow.log_metrics(metrics)
        
        return metrics
    
    def check_for_drift(self, reference_data, current_data):
        """Check for data drift and alert if necessary"""
        drift_report = self.calculate_data_drift(reference_data, current_data)
        
        if drift_report['data_drift']['data_drift_score'] > self.drift_threshold:
            self._send_alert('Data drift detected!')
            return True
        return False
    
    def _send_alert(self, message):
        """Send alert through configured channels"""
        # Implement your alert mechanism here (email, Slack, etc.)
        print(f"ALERT: {message}")

# src/monitoring/performance_logger.py
import logging
from datetime import datetime

class PerformanceLogger:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            filename=f"{self.config['monitoring']['logging_path']}/model_performance.log",
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    
    def log_prediction(self, input_data, prediction, actual=None):
        """Log each prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input_data': input_data.to_dict() if hasattr(input_data, 'to_dict') else input_data,
            'prediction': prediction,
            'actual': actual
        }
        logging.info(f"Prediction: {log_entry}")