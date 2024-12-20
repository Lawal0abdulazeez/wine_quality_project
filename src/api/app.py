# src/api/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys
import os

# Set up base path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)
from src.data_processing.preprocessor import DataPreprocessor

app = Flask(__name__)

# Load model and preprocessor using absolute paths
try:
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'saved_models', 'random_forest_model.pkl'))
    preprocessor_path = os.path.join(BASE_DIR, 'models', 'saved_models', 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        scaler_path = os.path.join(BASE_DIR, 'models', 'saved_models', 'scaler.pkl')
        preprocessor.scaler = joblib.load(scaler_path)
    else:
        from src.data_processing.preprocessor import DataPreprocessor
        config = {
            'raw_data_path': os.path.join(BASE_DIR, 'data', 'raw'),
            'model_save_path': os.path.join(BASE_DIR, 'models', 'saved_models'),
            'target_column': 'quality',
            'test_size': 0.2,
            'random_state': 42
        }
        preprocessor = DataPreprocessor(config)
        # Ensure the preprocessor is properly initialized and fitted
        data = preprocessor.load_data()
        preprocessor.preprocess(data)
        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump(preprocessor.scaler, scaler_path)
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for files in: {os.path.join(BASE_DIR, 'models', 'saved_models')}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess data
        processed_data = preprocessor.scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        # Load new data
        new_data = request.get_json()
        
        # Retrain model
        # Implementation for model retraining
        
        return jsonify({'status': 'success'})
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)