"""
Script to export trained model and scaler from the notebook
Run this script after training your model in the notebook
"""

import os
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

def export_models(model, scaler, output_dir='models'):
    """
    Save trained model and scaler to pickle files
    
    Parameters:
    -----------
    model : scikit-learn model object
        The trained model (e.g., LassoCV)
    scaler : StandardScaler object
        The fitted scaler used for feature standardization
    output_dir : str
        Directory to save the models (default: 'models')
    """
    
    # Create models directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save model
    model_path = os.path.join(output_dir, 'lasso_cv_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to: {model_path}")
    
    print("\n" + "="*50)
    print("Models exported successfully!")
    print("="*50)
    print("\nYou can now run the Flask app:")
    print("  python app.py")
    print("\nThen visit: http://localhost:5000")

if __name__ == '__main__':
    print("This script should be imported from your Jupyter notebook.")
    print("\nUsage in your notebook:")
    print("  from export_model import export_models")
    print("  export_models(lasso_cv, scaler)")
    print("\nWhere:")
    print("  - lasso_cv is your trained LassoCV model")
    print("  - scaler is your fitted StandardScaler")
