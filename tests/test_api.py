import pytest
from fastapi.testclient import TestClient

from api.main import app
from mmm.service import MMMService

client = TestClient(app=app)


@pytest.fixture(scope="session", autouse=True)
def train_model_once():
    """Train model once before running prediction tests."""
    service = MMMService(model_type="linear")
    service.train()


def test_train_endpoint():
    response = client.post("/train")

    if response.status_code != 200:
        raise AssertionError(f"Unexpected status code: {response.status_code}")

    data = response.json()
    if "message" not in data:
        raise AssertionError("Key 'message' missing in response JSON")

    if not data["message"].startswith("Model trained"):
        raise AssertionError(f"Unexpected message: {data['message']}")


def test_predict_endpoint():
    payload = [
        {
            "date": "2023-01-01",
            "tv_spend": 20000.0,
            "radio_spend": 5000.0,
            "social_media_spend": 3000.0,
            "search_spend": 4000.0,
            "print_spend": 2000.0,
            "outdoor_spend": 1000.0,
        }
    ]

    response = client.post("/predict", json=payload)

    if response.status_code != 200:
        raise AssertionError(f"Unexpected status code: {response.status_code}")

    data = response.json()
    if "predictions" not in data:
        raise AssertionError("Key 'predictions' missing in response JSON")

    predictions = data["predictions"]
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise AssertionError(f"Unexpected predictions format: {predictions}")

    prediction = predictions[0]

    if "date" not in prediction or not isinstance(prediction["date"], str):
        raise AssertionError("Prediction 'date' is missing or not a string")

    if "predicted_sales" not in prediction or not isinstance(
        prediction["predicted_sales"], float
    ):
        raise AssertionError("Prediction 'predicted_sales' is missing or not a float")
