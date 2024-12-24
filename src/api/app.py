# src/api/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys
import os
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data_processing.preprocessor import DataPreprocessor  # Updated import statement

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#from src.data_processing.preprocessor import generate_and_save_preprocessor



# Define the path to the preprocessor
preprocessor_path = os.path.join('models', 'saved_models', 'preprocessor.pkl')

# Check if the preprocessor file exists
if not os.path.exists(preprocessor_path):
    print("Preprocessor file not found. Generating it now...")
    generate_and_save_preprocessor()

# Load the preprocessor
preprocessor = joblib.load(preprocessor_path)
print("Preprocessor loaded successfully.")


app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('models/saved_models/random_forest_model.pkl')
preprocessor = joblib.load('models/saved_models/preprocessor.pkl')

@app.route('/')
def home():
    return "Hello, Flask!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Preprocess data
        processed_data = preprocessor.transform(df)  # Now, the scaler is already fitted

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
    #from src.api import app  # Ensure the script is run as a module
    app.run(host='0.0.0.0', port=5000)