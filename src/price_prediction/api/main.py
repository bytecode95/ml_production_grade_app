"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from price_prediction.api.exceptions import register_exception_handlers
from price_prediction.api.routes.health import router as health_router
from price_prediction.api.routes.model_info import router as model_info_router
from price_prediction.api.routes.predictions import router as predictions_router
from price_prediction.inference.predictor import HousePricePredictor
from price_prediction.observability.logging import configure_logging
from price_prediction.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize application dependencies and release resources."""
    configure_logging()

    artifact_path = Path(
        os.getenv(
            "MODEL_ARTIFACT_PATH",
            "artifacts/house_price_model.joblib",
        )
    )

    app.state.prediction_service = None

    try:
        predictor = HousePricePredictor(str(artifact_path))
        app.state.prediction_service = PredictionService(predictor)

        logger.info("Prediction service initialized.")
    except Exception:
        # Do not log sensitive details to the HTTP response.
        # Full exception details remain available in server logs.
        logger.exception(
            "Prediction service initialization failed. "
            "Readiness will remain unavailable."
        )

    yield

    app.state.prediction_service = None
    logger.info("Prediction service shut down.")


app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using a versioned trained ML model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(predictions_router, prefix="/api/v1")
app.include_router(model_info_router, prefix="/api/v1")

register_exception_handlers(app)