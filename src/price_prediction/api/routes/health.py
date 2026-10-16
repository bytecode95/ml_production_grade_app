"""Liveness and readiness endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Health"])


@router.get("/health/live")
def liveness() -> dict[str, str]:
    """Report that the process is alive."""
    return {"status": "alive"}


@router.get("/health/ready")
def readiness(request: Request) -> JSONResponse:
    """Report whether the model is ready for inference."""
    service = getattr(request.app.state, "prediction_service", None)

    if service is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "Prediction model is unavailable.",
            },
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready"},
    )