from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import numpy as np
import os
import uvicorn

app = FastAPI(title="Diabetes Prediction API")

model = Sequential([
    Dense(32, activation='relu', input_shape=(8,)),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.load_weights('diabetes.weights.h5')

scaler = joblib.load('scaler.pkl')

class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict_diabetes(data: PatientData):
    input_data = np.array([[
        data.Pregnancies, data.Glucose, data.BloodPressure, 
        data.SkinThickness, data.Insulin, data.BMI, 
        data.DiabetesPedigreeFunction, data.Age
    ]], dtype=np.float32)
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probability = float(prediction[0][0])
    
    return {"diabetes_probability": probability, "is_diabetic": bool(probability > 0.5)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
