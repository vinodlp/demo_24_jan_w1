from fastapi.testclient import TestClient
import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from main import app

client = TestClient(app)

def test__home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "welcome to fastapi application"

def test__predict_iris_endpoint():
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["Setosa", "Versicolor", "Virginica"]


def test_predict_endpoint_invalid_data():
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": "invalid"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422
