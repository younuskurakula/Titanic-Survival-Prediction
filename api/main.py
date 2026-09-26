from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "titanic_pipeline.joblib"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Machine Learning API for Titanic survival prediction",
    version="1.0.0"
)


class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float | None = None
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str | None = None


class PredictionResponse(BaseModel):
    prediction: int
    result: str
    survival_probability: float


@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: Passenger):

    passenger_data = pd.DataFrame([
        passenger.model_dump()
    ])

    prediction = model.predict(passenger_data)

    probability = model.predict_proba(
        passenger_data
    )

    if prediction[0] == 1:
        result = "Survived"
    else:
        result = "Did Not Survive"

    return PredictionResponse(
        prediction=int(prediction[0]),
        result=result,
        survival_probability=round(
            float(probability[0][1]),
            4
        )
    )