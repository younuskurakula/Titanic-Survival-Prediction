from pathlib import Path

import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "titanic_pipeline.joblib"


# --------------------------------------------------
# 1. Load Saved Pipeline
# --------------------------------------------------

model_pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# 2. Create New Passenger
# --------------------------------------------------

passengers = pd.DataFrame([
    {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100.0,
        "Embarked": "C"
    },
    {
        "Pclass": 3,
        "Sex": "male",
        "Age": 35,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 10.0,
        "Embarked": "S"
    }
])


print("\nPassenger data:")
print(passengers)


# --------------------------------------------------
# 3. Make Prediction
# --------------------------------------------------

prediction = model_pipeline.predict(passengers)
print("\nPrediction:", prediction)

# --------------------------------------------------
# 4. Get Prediction Probability
# --------------------------------------------------

probability = model_pipeline.predict_proba(passengers)


# --------------------------------------------------
# 5. Display Result
# --------------------------------------------------

if prediction[0] == 1:
    result = "Survived"
else:
    result = "Did Not Survive"


print("\nPrediction:", result)

print(
    "Probability of Not Surviving:",
    round(probability[0][0], 4)
)

print(
    "Probability of Surviving:",
    round(probability[0][1], 4)
)