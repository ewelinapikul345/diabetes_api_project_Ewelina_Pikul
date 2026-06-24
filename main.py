from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import os
import uvicorn

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
    tensor_input = tf.convert_to_tensor(scaled_data)
    
    input_key = list(infer.structured_input_signature[1].keys())[0]
    prediction = infer(**{input_key: tensor_input})
    
    probability = float(list(prediction.values())[0].numpy()[0][0])
    
    return {
        "diabetes_probability": probability, 
        "is_diabetic": bool(probability > 0.5)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
