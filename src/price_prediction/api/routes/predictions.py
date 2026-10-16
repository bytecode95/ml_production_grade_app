"""Prediction endpoint."""

import logging

from fastapi import APIRouter, Depends, status

from price_prediction.api.dependencies import get_prediction_service
from price_prediction.api.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from price_prediction.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict house prices",
)
def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Predict prices for one or more houses."""
    records = [
        house.model_dump(by_alias=True)
        for house in request.houses
    ]

    result = service.predict(records)

    return PredictionResponse.model_validate(result)