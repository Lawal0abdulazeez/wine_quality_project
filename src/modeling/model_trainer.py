import os
import json
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from src.data_processing.preprocessor import generate_and_save_preprocessor

# Generate and save the preprocessor before training
generate_and_save_preprocessor()

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'svm': SVC()
        }
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models, save them, and log performance using MLflow"""
        results = {}
        metrics_path = 'metrics/training_metrics.json'
        # Ensure the output directory exists
        os.makedirs('models/saved_models', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)

        for name, model in self.models.items():
            # Start MLflow run
            with mlflow.start_run(run_name=name):
                # Train model
                model.fit(X_train, y_train)
                
                # Predict and evaluate performance
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Log model parameters and performance metrics
                mlflow.log_params(model.get_params())
                mlflow.log_metric('accuracy', accuracy)
                
                # Save the model
                model_save_path = self.save_model(model, name)
                print(f"Model {name} saved to {model_save_path}")
                
                # Store only the model's performance in results (not the model object itself)
                results[name] = {
                    'model_path': model_save_path,
                    'accuracy': accuracy
                }

        # Save metrics to JSON file
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)  # Now `results` is JSON-serializable
        print(f"Metrics saved to {metrics_path}")
        
        return results

    def save_model(self, model, model_name):
        """Save the trained model to a file"""
        # Save the model using joblib
        model_save_path = f'models/saved_models/{model_name}_model.pkl'
        joblib.dump(model, model_save_path)
        return model_save_path

    def get_best_model(self, results):
        """Return the model with the highest accuracy"""
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model_path = results[best_model_name]['model_path']
        return best_model_path, best_model_name

    def hyperparameter_tuning(self, best_model_name, X_train, y_train):
        """Perform hyperparameter tuning for the best model"""
        param_distributions = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Create RandomizedSearchCV object
        random_search = RandomizedSearchCV(
            self.models[best_model_name],
            param_distributions[best_model_name],
            n_iter=10,
            cv=5,
            random_state=42
        )
        
        # Perform search
        random_search.fit(X_train, y_train)
        
        return random_search.best_estimator_

# Example usage:
if __name__ == "__main__":
    # Load your data (assuming you have X_train, X_test, y_train, y_test)
    from sklearn.datasets import load_wine
    data = load_wine()
    X = data.data
    y = data.target

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the trainer and train models
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Get the best model
    best_model_path, best_model_name = trainer.get_best_model(results)
    print(f"The best model is {best_model_name} with accuracy: {results[best_model_name]['accuracy']}")
    print(f"The best model is saved at {best_model_path}")
