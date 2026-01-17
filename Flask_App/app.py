import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load models
# Try loading from multiple possible locations
possible_paths = [
    os.path.join('models', 'lasso_cv_model.pkl'),  # Flask_App/models
    os.path.join('..', 'lasso_cv_model.pkl'),       # Parent directory (ML_Basics)
]
possible_scaler_paths = [
    os.path.join('models', 'scaler.pkl'),           # Flask_App/models
    os.path.join('..', 'scaler.pkl'),               # Parent directory (ML_Basics)
]

scaler = None
model = None
models_loaded = False

# Try each path until we find the models
for model_path, scaler_path in zip(possible_paths, possible_scaler_paths):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Verify model is an instance, not a class
        if hasattr(model, 'predict') and callable(model.predict):
            models_loaded = True
            print(f"✓ Models loaded from: {os.path.abspath(model_path)}")
            print(f"✓ Model type: {type(model).__name__}")
            break
        else:
            print(f"⚠ Model is not valid (likely a class instead of instance)")
            model = None
            scaler = None
    except FileNotFoundError:
        continue

if not models_loaded:
    print("Warning: Model files not found. Please train and save the model first.")

# Feature names (from your notebook)
FEATURE_NAMES = ['Region', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Rain', 
                 'Temperature', 'RH', 'Ws']

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions"""
    try:
        if not models_loaded:
            return jsonify({'error': 'Models not loaded. Please train the model first.'}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features in the correct order (9 features after correlation filtering)
        # Order: Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes (0 for no fire/low), Region
        features = np.array([[
            float(data.get('Temperature', 0)),
            float(data.get('RH', 0)),
            float(data.get('Ws', 0)),
            float(data.get('Rain', 0)),
            float(data.get('FFMC', 0)),
            float(data.get('DMC', 0)),
            float(data.get('ISI', 0)),
            0,  # Classes (0 for prediction, as we're predicting FWI not Classes)
            float(data.get('Region', 0))
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_value = round(float(prediction), 2)
        
        # Determine fire risk level based on FWI value
        if prediction_value < 5:
            risk_level = "Very Low"
        elif prediction_value < 15:
            risk_level = "Low"
        elif prediction_value < 30:
            risk_level = "Moderate"
        elif prediction_value < 50:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        return jsonify({
            'success': True,
            'prediction': prediction_value,
            'fwi_value': prediction_value,
            'risk_level': risk_level,
            'message': f'🔥 Predicted Fire Weather Index (FWI): {prediction_value}',
            'risk_message': f'Risk Level: {risk_level}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/features')
def get_features():
    """Get feature information"""
    feature_info = {
        'Region': {'type': 'integer', 'min': 0, 'max': 1, 'description': 'Region (0 or 1)'},
        'FFMC': {'type': 'float', 'min': 0, 'max': 100, 'description': 'Fine Fuel Moisture Code'},
        'DMC': {'type': 'float', 'min': 0, 'max': 300, 'description': 'Duff Moisture Code'},
        'DC': {'type': 'float', 'min': 0, 'max': 1000, 'description': 'Drought Code'},
        'ISI': {'type': 'float', 'min': 0, 'max': 60, 'description': 'Initial Spread Index'},
        'BUI': {'type': 'float', 'min': 0, 'max': 300, 'description': 'Buildup Index'},
        'FWI': {'type': 'float', 'min': 0, 'max': 100, 'description': 'Fire Weather Index'},
        'Rain': {'type': 'float', 'min': 0, 'max': 100, 'description': 'Rainfall (mm)'},
        'Temperature': {'type': 'integer', 'min': 2, 'max': 40, 'description': 'Temperature (°C)'},
        'RH': {'type': 'integer', 'min': 15, 'max': 100, 'description': 'Relative Humidity (%)'},
        'Ws': {'type': 'integer', 'min': 0, 'max': 40, 'description': 'Wind Speed (km/h)'}
    }
    return jsonify(feature_info)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
