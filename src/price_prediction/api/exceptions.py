"""API exception types and handlers."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Expected application-level prediction failure."""


def register_exception_handlers(app: FastAPI) -> None:
    """Register consistent, safe error responses."""

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "The request does not match the API schema.",
                }
            },
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(
        request: Request,
        exc: PredictionError,
    ) -> JSONResponse:
        logger.warning("Prediction request failed: %s", exc)
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "PREDICTION_ERROR",
                    "message": "The supplied data could not be used for prediction.",
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception("Unhandled API error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected server error occurred.",
                }
            },
        )