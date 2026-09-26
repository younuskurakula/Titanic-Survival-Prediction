import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def load_data():

    # Load dataset
    train = pd.read_csv('/Users/kurakulaenus/A - Python/B - AIMLGenAIAgenticAI/ZB - Real World Projects/BKaggleProjects/ATitanicSurvivalPrediction/data/train.csv')
    data = train.copy()

    X = data.drop("Survived", axis=1)
    y = data["Survived"]

    columns_to_drop = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    X = X.drop(columns=columns_to_drop)

    return X, y


def create_preprocessor():

    numerical_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    categorical_features = [
        "Sex",
        "Embarked"
    ]

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor


def create_model_pipeline():

    preprocessor = create_preprocessor()

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )
    ])

    return pipeline