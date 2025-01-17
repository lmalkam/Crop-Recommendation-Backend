from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Define request structure
class PredictionRequest(BaseModel):
    features: list

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; restrict to ["http://localhost:3000"] for stricter policy
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.post('/predict')
def predict(request: PredictionRequest):
    # Convert input data to NumPy array
    features = np.array(request.features).reshape(1, -1)
    # Make prediction
    prediction = model.predict(features)
    # Return prediction
    return {'prediction': prediction.tolist()}

@app.get('/health')
def health():
    return f"Healthy"

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
