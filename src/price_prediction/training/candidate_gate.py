
"""Accept or reject candidates using the configured model policy."""

import logging
from dataclasses import dataclass

from price_prediction.domain.model_policy import ModelPolicy
from price_prediction.training.evaluator import CandidateEvaluation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateDecision:
    """Acceptance decision for one evaluated candidate."""

    name: str
    accepted: bool
    reasons: tuple[str, ...]
    evaluation: CandidateEvaluation


def assess_candidate(
    evaluation: CandidateEvaluation,
    policy: ModelPolicy,
) -> CandidateDecision:
    """Apply acceptance thresholds to a candidate's validation metrics."""
    failures = policy.check(evaluation.metrics)

    decision = CandidateDecision(
        name=evaluation.name,
        accepted=not failures,
        reasons=tuple(failures),
        evaluation=evaluation,
    )

    if decision.accepted:
        logger.info(
            "Candidate accepted: name=%s metrics=%s",
            decision.name,
            evaluation.metrics.to_dict(),
        )
    else:
        logger.warning(
            "Candidate rejected: name=%s reasons=%s",
            decision.name,
            decision.reasons,
        )

    return decision


def select_best_candidate(
    evaluations: list[CandidateEvaluation],
    policy: ModelPolicy,
    *,
    ranking_metric: str = "mae",
) -> CandidateDecision:
    """Select the best candidate that satisfies the policy.

    Selection is based exclusively on validation results.
    """
    if not evaluations:
        raise ValueError("No candidate evaluations were provided.")

    if ranking_metric not in {"mae", "rmse", "mape", "r2"}:
        raise ValueError(
            "ranking_metric must be one of: mae, rmse, mape, r2."
        )

    decisions = [
        assess_candidate(evaluation, policy)
        for evaluation in evaluations
    ]

    accepted = [decision for decision in decisions if decision.accepted]

    if not accepted:
        reasons = {
            decision.name: decision.reasons
            for decision in decisions
        }
        logger.error("No acceptable candidate found: %s", reasons)
        raise RuntimeError(
            f"No candidate satisfied the model policy. Rejections: {reasons}"
        )

    def ranking_key(decision: CandidateDecision):
        value = getattr(decision.evaluation.metrics, ranking_metric)

        if value is None:
            return (1, 0.0)

        score = -value if ranking_metric == "r2" else value
        return (0, score)

    best = min(accepted, key=ranking_key)

    logger.info(
        "Best candidate selected: name=%s ranking_metric=%s value=%s",
        best.name,
        ranking_metric,
        getattr(best.evaluation.metrics, ranking_metric),
    )

    return best
