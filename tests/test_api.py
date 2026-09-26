from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_prediction():

    passenger = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "C"
    }

    response = client.post(
        "/predict",
        json=passenger
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "result" in result
    assert "survival_probability" in result


def test_invalid_input():

    passenger = {
        "Pclass": "invalid",
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "C"
    }

    response = client.post(
        "/predict",
        json=passenger
    )

    assert response.status_code == 422