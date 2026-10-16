"""Model information endpoint."""

from fastapi import APIRouter, Depends

from price_prediction.api.dependencies import get_prediction_service
from price_prediction.api.schemas.prediction import ModelInfoResponse
from price_prediction.services.prediction_service import PredictionService

router = APIRouter(tags=["Model"])


@router.get(
    "/model",
    response_model=ModelInfoResponse,
    summary="Get active model metadata",
)
def model_info(
    service: PredictionService = Depends(get_prediction_service),
) -> ModelInfoResponse:
    """Return the currently loaded model's public metadata."""
    return ModelInfoResponse.model_validate(service.get_model_info())