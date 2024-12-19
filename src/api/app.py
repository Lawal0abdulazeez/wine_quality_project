# src/api/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.data_processing.preprocessor import DataPreprocessor

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('models/saved_models/best_model.pkl')
preprocessor = joblib.load('models/saved_models/preprocessor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess data
        processed_data = preprocessor.transform(df)
        
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