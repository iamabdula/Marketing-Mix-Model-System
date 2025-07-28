import pandas as pd

from mmm.service import MMMService


def test_service_train_and_predict(tmp_path):
    model_dir = tmp_path / "models"
    service = MMMService(model_type="linear", model_dir=model_dir)

    service.train()

    model_path = model_dir / "linear_mmm_model.pkl"
    if not model_path.exists():
        raise AssertionError("Model file was not created after training.")

    service.load_model()

    new_data = pd.DataFrame(
        {
            "date": ["2023-01-01"],
            "tv_spend": [20000.0],
            "radio_spend": [5000.0],
            "social_media_spend": [3000.0],
            "search_spend": [4000.0],
            "print_spend": [2000.0],
            "outdoor_spend": [1000.0],
        }
    )

    preds = service.predict(new_data)

    if not isinstance(preds, pd.Series):
        raise AssertionError("Predictions should be a pandas Series.")
    if len(preds) != 1:
        raise AssertionError("Predictions should contain exactly 1 value.")
    if preds.isna().any():
        raise AssertionError("Predictions contain NaN values.")
