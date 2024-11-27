
import joblib
import os

#path to the scaler
SCALER_PATH = os.path.join(os.getcwd(), 'model', 'scaler.pkl')

def load_scaler():
    try:
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded successfully.")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

