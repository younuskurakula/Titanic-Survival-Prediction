import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from train import model_pipeline


# Load dataset
train = pd.read_csv('/Users/kurakulaenus/A - Python/B - AIMLGenAIAgenticAI/ZB - Real World Projects/BKaggleProjects/ATitanicSurvivalPrediction/data/train.csv')
data = train.copy()


X = data.drop("Survived", axis=1)
y = data["Survived"]


# Drop unnecessary columns
columns_to_drop = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

X = X.drop(columns=columns_to_drop)


# Create Stratified K-Fold
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Metrics
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# Cross-validation
scores = cross_validate(
    model_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring
)


# Display results
print("Cross-Validation Results")
print("------------------------")

for metric in scoring:
    values = scores[f"test_{metric}"]

    print(
        f"{metric.capitalize()}: "
        f"{values.mean():.4f} "
        f"+/- {values.std():.4f}"
    )