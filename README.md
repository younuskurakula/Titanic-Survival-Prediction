# 🚢 Titanic Survival Prediction

An end-to-end machine learning classification project based on the **Kaggle Titanic: Machine Learning from Disaster** competition.

The project predicts whether a passenger survived the Titanic disaster using passenger information such as class, gender, age, family size, fare, and port of embarkation.

The project was developed from **data preprocessing and model training to model evaluation, API development, automated testing, Docker containerization, and cloud deployment**.

---

## 🎯 Objective

Build a complete machine learning system that can:

* Analyze and preprocess Titanic passenger data
* Handle missing values and categorical features
* Train and evaluate a classification model
* Apply cross-validation and hyperparameter tuning
* Save the trained ML pipeline
* Serve predictions through a REST API
* Validate API inputs
* Test the API automatically
* Run the application inside a Docker container
* Provide a simple web interface for predictions

---

## 📊 Dataset

**Source:** Kaggle — Titanic: Machine Learning from Disaster

**Problem Type:** Supervised Learning → Binary Classification

**Target Variable:** `Survived`

| Value | Meaning         |
| ----- | --------------- |
| `0`   | Did Not Survive |
| `1`   | Survived        |

### Dataset Files

* `train.csv` — Training dataset containing passenger features and the target variable
* `test.csv` — Kaggle test dataset used for generating predictions

### Features Used

* `Pclass` — Passenger class
* `Sex` — Passenger gender
* `Age` — Passenger age
* `SibSp` — Number of siblings or spouses aboard
* `Parch` — Number of parents or children aboard
* `Fare` — Passenger fare
* `Embarked` — Port of embarkation

The columns `PassengerId`, `Name`, `Ticket`, and `Cabin` are excluded from the model pipeline.

---

## 🔄 Machine Learning Workflow

The project follows an end-to-end ML workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Train / Validation Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Model Training
   ↓
Model Serialization
   ↓
FastAPI Prediction API
   ↓
API Testing
   ↓
Docker
   ↓
Cloud Deployment
```

---

## 🔍 Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline to keep training and inference preprocessing consistent.

### Numerical Features

The following numerical features are processed using median imputation:

```text
Pclass
Age
SibSp
Parch
Fare
```

### Categorical Features

The following categorical features are processed using:

* Most-frequent-value imputation
* One-hot encoding
* Unknown-category handling

```text
Sex
Embarked
```

### Why Use a Pipeline?

The preprocessing steps and machine learning model are combined into a single Scikit-learn Pipeline.

This allows the same preprocessing logic to be automatically applied during both:

```text
Training
   ↓
Preprocessing
   ↓
Model
```

and:

```text
API Request
   ↓
Same Preprocessing
   ↓
Model
   ↓
Prediction
```

This reduces the risk of training and inference preprocessing mismatch.

---

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** from Scikit-learn.

Random Forest is an ensemble classification algorithm that combines multiple decision trees to make predictions.

The model is trained inside the complete preprocessing pipeline.

### Model Development

The project includes:

* Train/validation split
* Stratified sampling
* Cross-validation
* Accuracy evaluation
* Precision
* Recall
* F1-score
* Hyperparameter tuning

The final trained pipeline is saved using **Joblib**:

```text
models/titanic_pipeline.joblib
```

---

## 📈 Results

### Local Validation Performance

**Accuracy:** approximately **80%**

### Kaggle Performance

**Kaggle Score:** approximately **74%**

The difference between local validation performance and Kaggle performance can occur because the local validation set and Kaggle's hidden test data are different datasets.

These results represent the current version of the project and may change after further model tuning or feature engineering.

---

## 🚀 Prediction API

The trained model is exposed through a **FastAPI REST API**.

### API Endpoints

#### Health Check

```http
GET /health
```

Example response:

```json
{
    "status": "healthy"
}
```

#### Prediction

```http
POST /predict
```

Example request:

```json
{
    "Pclass": 1,
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100,
    "Embarked": "C"
}
```

Example response:

```json
{
    "prediction": 1,
    "result": "Survived",
    "survival_probability": 0.95
}
```

#### API Documentation

FastAPI automatically provides interactive API documentation through:

```text
/docs
```

---

## 🌐 Simple Web Interface

The project includes a lightweight HTML and JavaScript frontend.

The user can enter passenger information through a simple form:

```text
Passenger Information
        ↓
HTML Form
        ↓
JavaScript
        ↓
POST /predict
        ↓
FastAPI
        ↓
ML Pipeline
        ↓
Prediction
        ↓
Result displayed on webpage
```

No frontend framework is required for this application.

---

## 🧪 API Testing

The API is tested using **Pytest** and FastAPI's `TestClient`.

Current tests cover:

* Health endpoint
* Valid prediction request
* Invalid input validation

Run the tests with:

```bash
python -m pytest
```

Expected result:

```text
3 passed
```

---

## 🐳 Docker

The application is containerized using Docker.

### Build Docker Image

```bash
docker build -t titanic-ml-api .
```

### Run Container

```bash
docker run -p 8000:8000 titanic-ml-api
```

The application can then be accessed at:

```text
http://localhost:8000
```

### Docker Compose

The project also includes Docker Compose configuration for easier application management.

Start the application:

```bash
docker compose up --build
```

Run in background:

```bash
docker compose up --build -d
```

Stop the application:

```bash
docker compose down
```

---

## ☁️ Deployment

The application is designed for cloud deployment using the Dockerized FastAPI application.

Deployment architecture:

```text
GitHub Repository
       ↓
Dockerfile
       ↓
Container Build
       ↓
Cloud Platform
       ↓
Public FastAPI Application
       ↓
Web Frontend + ML API
```

The deployed application provides:

```text
/
    → Web Interface

/health
    → Health Check

/predict
    → ML Prediction

/docs
    → Swagger API Documentation
```

---

## 🛠️ Technology Stack

### Programming

* Python
* HTML
* JavaScript

### Data Science and Machine Learning

* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

### Machine Learning

* Random Forest
* Classification
* Feature preprocessing
* Missing value imputation
* One-hot encoding
* Cross-validation
* Hyperparameter tuning
* Model evaluation

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Testing

* Pytest
* FastAPI TestClient
* HTTPX

### Deployment and DevOps

* Docker
* Docker Compose
* Git
* GitHub
* Cloud deployment

---

## 📁 Project Structure

```text
ATitanicSurvivalPrediction/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── models/
│   └── titanic_pipeline.joblib
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── predict.py
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── frontend/
│   └── index.html
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/younuskurakula/Titanic-Survival-Prediction.git
```

```bash
cd Titanic-Survival-Prediction
```

### 2. Create and activate environment

```bash
python3 -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

### 5. Open the application

```text
http://127.0.0.1:8000
```

### 6. Open API documentation

```text
http://127.0.0.1:8000/docs
```

---

## 🐳 Run with Docker

Build:

```bash
docker build -t titanic-ml-api .
```

Run:

```bash
docker run -p 8000:8000 titanic-ml-api
```

Open:

```text
http://localhost:8000
```

---

## 🧪 Run Tests

```bash
python -m pytest
```

---

## 🎓 Key Learning Outcomes

This project helped demonstrate the complete lifecycle of a machine learning application:

* Exploratory Data Analysis
* Data preprocessing
* Feature engineering
* Feature selection
* Model training
* Model evaluation
* Cross-validation
* Hyperparameter tuning
* Model serialization
* Batch and single-record inference
* REST API development
* Input validation
* Automated testing
* Docker containerization
* Frontend and backend integration
* Git and GitHub
* Cloud deployment

The main focus was moving beyond a Jupyter Notebook and building a **deployable machine learning application**.

---

## 🔮 Future Improvements

Potential improvements include:

* Additional feature engineering
* Model comparison
* Advanced hyperparameter optimization
* CI/CD using GitHub Actions
* Experiment tracking with MLflow
* Dataset and model versioning
* API logging and monitoring
* Model performance monitoring
* Automated model retraining

---

## 👤 Author

**Younus Kurakula**

[GitHub](https://github.com/younuskurakula) · [LinkedIn](https://www.linkedin.com/in/enus5three/)
