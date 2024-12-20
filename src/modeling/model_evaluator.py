
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
