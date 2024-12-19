# src/modeling/model_trainer.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'svm': SVC()
        }
        
    def train_models(self, X_train, y_train):
        """Train multiple models and return results"""
        results = {}
        
        for name, model in self.models.items():
            # Start MLflow run
            with mlflow.start_run(run_name=name):
                # Train model
                model.fit(X_train, y_train)
                
                # Log model parameters
                mlflow.log_params(model.get_params())
                
                # Save model
                results[name] = model
                
        return results

    def hyperparameter_tuning(self, best_model_name, X_train, y_train):
        """Perform hyperparameter tuning for the best model"""
        from sklearn.model_selection import RandomizedSearchCV
        
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

# src/modeling/model_evaluator.py
class ModelEvaluator:
    @staticmethod
    def evaluate_models(models, X_test, y_test):
        """Evaluate multiple models and return results"""
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
            
        return results