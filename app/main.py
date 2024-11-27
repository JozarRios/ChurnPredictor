
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from app.encoder_loader import load_encoder
from app.model_loader import load_model
from app.scaler_loader import load_scaler

#innitialize FastAPI app
app = FastAPI()

#we load the model, scaler, encoder
model = load_model()
if model is None:
    raise Exception("Model not loaded successfully")

scaler = load_scaler()
if scaler is None:
    raise Exception("Scaler not loaded")

encoder = load_encoder()
if encoder is None:
    raise Exception("Encoder not loaded")



#input data schema - Pydantic model
class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
   

#root endpoint for the API
@app.get("/")
def read_root():
    """root endpoint for the API."""
    return{"message":"Welcome to the Churn Prediction API"}

#endpoint for predictions
@app.post("/predict/")
def predict_churn(customer: CustomerData):
    """Predict customer churn based on input data."""
    #converts the input to pandas DataFrame
    input_data = pd.DataFrame([customer.dict()])

    # Preprocess input data
    input_data["Gender"] = encoder.transform(input_data["Gender"])  # Encode Gender
    input_data = pd.get_dummies(input_data, columns=["Geography"], drop_first=True)  # One-hot encode Geography

    # Add missing columns for Geography
    for col in [col for col in model.feature_names_in_ if col.startswith("Geography")]:
        if col not in input_data.columns:
            input_data[col] = 0

    # Scale numerical features
    numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    #ensure all expected columns are present
    expected_columns = [
        "CreditScore", "Geography_Germany", "Geography_Spain", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard","IsActiveMember", "EstimatedSalary"
    ]
    # Add missing columns for Geography (dynamic handling)
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill missing columns with default value 0

    #reorder columns to match training data
    input_data = input_data[model.feature_names_in_]

    #make a prediction
    try:
        prediction = model.predict(input_data)
        prediction_prob = model.predict_proba(input_data)[:, 1]  # Probability of churn
        return {
            "prediction": int(prediction[0]),  # 0 for no churn, 1 for churn
            "probability": float(prediction_prob[0])  # Probability of churn
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))