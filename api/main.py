import pandas as pd
from fastapi import FastAPI

from api.schemas import PredictionOutput, PredictionResponse, SpendInput
from mmm.service import MMMService

app = FastAPI(
    title="Marketing Mix Model API",
    description="API for training and predicting sales using MMM.",
    version="1.0.0",
)

# Initialize service
service = MMMService(model_type="linear")


@app.post("/train", summary="Train the MMM model")
def train_model():
    """
    Retrain the MMM model using processed data.
    """
    service.train()
    return {"message": "Model trained and saved successfully."}


@app.post("/predict", response_model=PredictionResponse, summary="Predict sales")
def predict_sales(data: list[SpendInput]):
    """
    Predict sales based on marketing spend input.
    """
    # Convert input to DataFrame
    new_data = pd.DataFrame([d.model_dump() for d in data])
    service.load_model()  # Ensure model is loaded
    preds = service.predict(new_data)

    # Prepare response
    predictions = [
        PredictionOutput(date=row["date"], predicted_sales=pred)
        for row, pred in zip(new_data.to_dict(orient="records"), preds)
    ]

    return PredictionResponse(predictions=predictions)
