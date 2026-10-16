
"""Evaluate trained regression candidates on validation data."""

import logging
from dataclasses import dataclass

import pandas as pd

from price_prediction.domain.metrics import (
    RegressionMetrics,
    calculate_metrics,
)
from price_prediction.training.trainer import TrainedCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Validation results for a trained candidate."""

    name: str
    model: object
    fit_seconds: float
    metrics: RegressionMetrics


def evaluate_candidate(
    candidate: TrainedCandidate,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> CandidateEvaluation:
    """Evaluate one fitted candidate without refitting it."""

    if X_val.empty:
        raise ValueError("Validation features cannot be empty.")

    if len(X_val) != len(y_val):
        raise ValueError(
            "X_val and y_val must contain the same number of rows."
        )

    if y_val.isna().any():
        raise ValueError("Validation target contains missing values.")

    logger.info(
        "Candidate evaluation started: name=%s rows=%d",
        candidate.name,
        len(X_val),
    )

    try:
        predictions = candidate.model.predict(X_val)
        metrics = calculate_metrics(y_val, predictions)
    except Exception:
        logger.exception(
            "Candidate evaluation failed: name=%s",
            candidate.name,
        )
        raise

    result = CandidateEvaluation(
        name=candidate.name,
        model=candidate.model,
        fit_seconds=candidate.fit_seconds,
        metrics=metrics,
    )

    logger.info(
        "Candidate evaluation completed: name=%s metrics=%s",
        result.name,
        result.metrics.to_dict(),
    )

    return result


def evaluate_candidates(
    candidates: list[TrainedCandidate],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    ranking_metric: str = "mae",
) -> list[CandidateEvaluation]:
    """Evaluate candidates and rank them using validation metrics."""

    if not candidates:
        raise ValueError("At least one trained candidate is required.")

    if ranking_metric not in {"mae", "rmse", "mape", "r2"}:
        raise ValueError(
            "ranking_metric must be one of: mae, rmse, mape, r2."
        )

    results = [
        evaluate_candidate(candidate, X_val, y_val)
        for candidate in candidates
    ]

    def ranking_key(result: CandidateEvaluation):
        value = getattr(result.metrics, ranking_metric)

        # Undefined MAPE ranks after candidates with valid MAPE.
        if value is None:
            return (1, 0.0)

        # Lower errors are better; higher R² is better.
        score = -value if ranking_metric == "r2" else value
        return (0, score)

    results.sort(key=ranking_key)

    logger.info(
        "Candidate ranking completed: metric=%s ranking=%s",
        ranking_metric,
        [
            {
                "name": result.name,
                "value": getattr(result.metrics, ranking_metric),
            }
            for result in results
        ],
    )

    return results
