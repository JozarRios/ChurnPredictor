
import joblib
import os

#path to the model
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'ChurnPredictionModel_v.0.0.1.pkl')

def load_model():
    """Load the churn prediction model"""
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

