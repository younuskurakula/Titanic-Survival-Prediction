from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib


from preprocessing import load_data, create_preprocessor


# Load data
X, y = load_data()


# Create pipeline
pipeline = Pipeline([
    (
        "preprocessor",
        create_preprocessor()
    ),
    (
        "model",
        RandomForestClassifier(
            random_state=42
        )
    )
])


# Hyperparameter grid
param_grid = {

    "model__n_estimators": [
        50,
        100,
        200
    ],

    "model__max_depth": [
        5,
        10,
        15,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ]
}


# Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)


print("Starting GridSearchCV...")

grid_search.fit(X, y)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)



# Save the best pipeline
joblib.dump(
    grid_search.best_estimator_,
    "models/titanic_pipeline.joblib"
)

print("\nFinal tuned model saved successfully.")