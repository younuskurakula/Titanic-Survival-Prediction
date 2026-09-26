# Import Basic Libraries
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

train = pd.read_csv('/Users/kurakulaenus/A - Python/B - AIMLGenAIAgenticAI/ZB - Real World Projects/BKaggleProjects/ATitanicSurvivalPrediction/data/train.csv')
data = train.copy()

print("Dataset loaded successfully.")
print("Shape:", data.shape)


# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

X = data.drop("Survived", axis=1) # - Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
y = data["Survived"] # - Survived

print("Feature of X")
print(X.head())
print("\nTarget of y")
print(y.head())

# --------------------------------------------------
# 3. Remove Unnecessary Columns
# --------------------------------------------------

columns_to_drop = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

X = X.drop(columns=columns_to_drop)


# --------------------------------------------------
# 4. Define Feature Types
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Numerical Preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# --------------------------------------------------
# 6. Categorical Preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# --------------------------------------------------
# 7. Combine Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 8. Create Complete ML Pipeline
# --------------------------------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])


# --------------------------------------------------
# 9. Train / Validation Split
# --------------------------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training samples:", X_train.shape[0])
print("Validation samples:", X_valid.shape[0])


# --------------------------------------------------
# 10. Train Model
# --------------------------------------------------

model_pipeline.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 11. Validation Prediction
# --------------------------------------------------

y_pred = model_pipeline.predict(X_valid)


# --------------------------------------------------
# 12. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_valid, y_pred)

print("\nValidation Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_valid, y_pred))


# --------------------------------------------------
# 13. Train Final Model on All Data
# --------------------------------------------------

model_pipeline.fit(X, y)

print("\nFinal model trained on complete dataset.")


# --------------------------------------------------
# 14. Save Complete Pipeline
# --------------------------------------------------

joblib.dump(
    model_pipeline,
    "models/titanic_pipeline.joblib"
)

print("Pipeline saved successfully.")

# train - predict - cross_validation - preprocessing - evaluate - tune - main - test