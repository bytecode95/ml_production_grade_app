
"""Baseline regression model and validation benchmark."""

import logging

from price_prediction.domain.metrics import (
    RegressionMetrics,
    calculate_metrics,
)
from price_prediction.domain.schemas import FeatureSchema
from price_prediction.features.transformations import build_sklearn_pipeline
from price_prediction.training.model_factory import create_model

logger = logging.getLogger(__name__)


def evaluate_baseline(
    X_train,
    y_train,
    X_val,
    y_val,
    schema: FeatureSchema,
    *,
    reference_year: int = 2026,
) -> tuple[object, RegressionMetrics]:
    """Fit a baseline on training data and evaluate on validation data."""
    estimator = create_model("baseline")

    pipeline = build_sklearn_pipeline(
        schema=schema,
        model=estimator,
        reference_year=reference_year,
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)

    logger.info(
        "Baseline evaluated: metrics=%s",
        metrics.to_dict(),
    )

    return pipeline, metrics
