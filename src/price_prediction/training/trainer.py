
"""Train individual regression candidates."""

import logging
import time
from dataclasses import dataclass
from typing import Any

from sklearn.pipeline import Pipeline

from price_prediction.domain.schemas import FeatureSchema
from price_prediction.features.transformations import build_sklearn_pipeline
from price_prediction.training.model_factory import create_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainedCandidate:
    """A fitted model candidate and its training metadata."""

    name: str
    model: Pipeline
    fit_seconds: float


def train_candidate(
    name: str,
    X_train,
    y_train,
    schema: FeatureSchema,
    *,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
    reference_year: int = 2026,
) -> TrainedCandidate:
    """Create and fit one candidate using training data only."""
    logger.info(
        "Training candidate started: name=%s rows=%d features=%d",
        name,
        len(X_train),
        X_train.shape[1],
    )

    estimator = create_model(
        name,
        random_state=random_state,
        params=params,
    )

    pipeline = build_sklearn_pipeline(
        schema=schema,
        model=estimator,
        reference_year=reference_year,
    )

    start_time = time.perf_counter()

    try:
        pipeline.fit(X_train, y_train)
    except Exception:
        logger.exception("Candidate training failed: name=%s", name)
        raise

    fit_seconds = time.perf_counter() - start_time

    logger.info(
        "Training candidate completed: name=%s fit_seconds=%.3f",
        name,
        fit_seconds,
    )

    return TrainedCandidate(
        name=name,
        model=pipeline,
        fit_seconds=fit_seconds,
    )
