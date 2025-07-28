from typing import List

from pydantic import BaseModel, Field


class SpendInput(BaseModel):
    date: str = Field(..., json_schema_extra={"example": "2023-01-01"})
    tv_spend: float = Field(..., json_schema_extra={"example": 20000})
    radio_spend: float = Field(..., json_schema_extra={"example": 5000})
    social_media_spend: float = Field(..., json_schema_extra={"example": 3000})
    search_spend: float = Field(..., json_schema_extra={"example": 4000})
    print_spend: float = Field(..., json_schema_extra={"example": 2000})
    outdoor_spend: float = Field(..., json_schema_extra={"example": 1000})


class PredictionOutput(BaseModel):
    date: str
    predicted_sales: float


class PredictionResponse(BaseModel):
    predictions: List[PredictionOutput]
