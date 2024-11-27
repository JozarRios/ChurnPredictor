
import joblib
import os

#path to the encoder
ENCODER_PATH = os.path.join(os.getcwd(), 'model', 'label_encoder.pkl')

def load_encoder():
    try:
        encoder = joblib.load(ENCODER_PATH)
        print("Encoder loaded successfully.")
        return encoder
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return None

