"""Centralized MLflow experiment tracking."""

import logging
import os
from contextlib import contextmanager
from collections.abc import Iterator, Mapping
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def configure_mlflow() -> None:
    """Configure the MLflow tracking destination and experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "ames-housing-price-prediction",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(
        "MLflow configured: tracking_uri=%s experiment=%s",
        tracking_uri,
        experiment_name,
    )


@contextmanager
def tracked_run(
    run_name: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[str]:
    """Create an MLflow run and log its initial parameters and tags."""
    configure_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        if parameters:
            # MLflow parameters must be scalar-like values.
            safe_parameters = {
                str(key): _parameter_value(value)
                for key, value in parameters.items()
            }
            mlflow.log_params(safe_parameters)

        if tags:
            mlflow.set_tags(dict(tags))

        logger.info("MLflow run started: name=%s id=%s", run_name, run_id)
        yield run_id


def log_metrics(metrics: Mapping[str, float | int | None]) -> None:
    """Log valid numeric metrics to the active MLflow run."""
    numeric_metrics: dict[str, float] = {}

    for name, value in metrics.items():
        if value is None:
            continue

        numeric_value = float(value)

        if not __import__("math").isfinite(numeric_value):
            raise ValueError(f"Metric {name!r} must be finite.")

        numeric_metrics[name] = numeric_value

    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)


def _parameter_value(value: Any) -> str | bool | int | float:
    """Convert parameters into MLflow-compatible scalar values."""
    if isinstance(value, (str, bool, int, float)):
        return value

    return str(value)