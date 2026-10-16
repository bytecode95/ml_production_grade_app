"""FastAPI dependencies."""

from fastapi import HTTPException, Request, status

from price_prediction.services.prediction_service import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    """Retrieve the initialized prediction service."""
    service = getattr(request.app.state, "prediction_service", None)

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_READY",
                "message": "Prediction service is not ready.",
            },
        )

    return service