import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import PredictionOutput, PredictionResponse, SpendInput
from mmm.service import MMMService

from .deps import get_service

router = APIRouter()


@router.post("/train", summary="Train the MMM model")
def train_model(service: MMMService = Depends(get_service)):
    try:
        service.train()
        return {"message": "Model trained and saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictionResponse, summary="Predict sales")
def predict_sales(data: list[SpendInput], service: MMMService = Depends(get_service)):
    try:
        new_data = pd.DataFrame([d.model_dump() for d in data])
        service.load_model()
        preds = service.predict(new_data)

        predictions = [
            PredictionOutput(date=row["date"], predicted_sales=float(pred))
            for row, pred in zip(new_data.to_dict(orient="records"), preds)
        ]

        return PredictionResponse(predictions=predictions)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Trained model not found. Train the model first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
