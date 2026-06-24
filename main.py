from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np

app = FastAPI(title="Diabetes Prediction API")
model = tf.saved_model.load('diabetes_model_saved')
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
    
    infer = model.signatures["serving_default"]
    prediction = infer(tf.convert_to_tensor(scaled_data))
    
    probability = float(list(prediction.values())[0].numpy()[0][0])
    
    return {"diabetes_probability": probability, "is_diabetic": bool(probability > 0.5)}
