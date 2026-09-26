from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import load_data, create_preprocessor


# Load data
X, y = load_data()


# Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Models
models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# Compare models
for name, model in models.items():

    pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "model",
            model
        )
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )

    print("\n", name)
    print("-" * 40)

    print(
        "Accuracy:",
        round(scores["test_accuracy"].mean(), 4)
    )

    print(
        "Precision:",
        round(scores["test_precision"].mean(), 4)
    )

    print(
        "Recall:",
        round(scores["test_recall"].mean(), 4)
    )

    print(
        "F1:",
        round(scores["test_f1"].mean(), 4)
    )