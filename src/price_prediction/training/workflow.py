
"""End-to-end model training, validation and final testing."""

import logging
from dataclasses import dataclass

import pandas as pd

from price_prediction.data.splitting import split_dataset
from price_prediction.domain.metrics import (
    RegressionMetrics,
    calculate_metrics,
)
from price_prediction.domain.model_policy import ModelPolicy
from price_prediction.domain.schemas import FeatureSchema
from price_prediction.training.baseline import evaluate_baseline
from price_prediction.training.candidate_gate import select_best_candidate
from price_prediction.training.evaluator import evaluate_candidates
from price_prediction.training.trainer import train_candidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingWorkflowResult:
    """Results produced by a completed training workflow."""

    selected_model_name: str
    selected_model: object
    validation_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    baseline_metrics: RegressionMetrics


def run_training_workflow(
    df: pd.DataFrame,
    schema: FeatureSchema,
    policy: ModelPolicy,
) -> TrainingWorkflowResult:
    """Train candidates, select one on validation data and test it once."""

    logger.info("Model training workflow started")

    # 1. Split the validated dataset.
    splits = split_dataset(df, target=schema.target)

    # 2. Establish the baseline using the same training/validation split.
    _, baseline_metrics = evaluate_baseline(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        schema,
    )

    logger.info(
        "Baseline validation metrics: %s",
        baseline_metrics.to_dict(),
    )

    # 3. Train candidate models on training data only.
    candidates = [
        train_candidate(
            name="random_forest",
            X_train=splits.X_train,
            y_train=splits.y_train,
            schema=schema,
            params={"n_estimators": 200, "max_depth": 15},
        ),
        train_candidate(
            name="extra_trees",
            X_train=splits.X_train,
            y_train=splits.y_train,
            schema=schema,
            params={"n_estimators": 200, "max_depth": 15},
        ),
    ]

    # 4. Compare candidates on the validation partition.
    evaluations = evaluate_candidates(
        candidates,
        splits.X_val,
        splits.y_val,
        ranking_metric="mae",
    )

    # 5. Select the best candidate satisfying the policy.
    decision = select_best_candidate(
        evaluations,
        policy,
        ranking_metric="mae",
    )

    selected = decision.evaluation

    logger.info(
        "Candidate selected: name=%s validation_metrics=%s",
        selected.name,
        selected.metrics.to_dict(),
    )

    # 6. Use the test set only for the selected candidate's
    # final evaluation. Do not refit or tune using test data.
    test_predictions = selected.model.predict(splits.X_test)

    test_metrics = calculate_metrics(
        splits.y_test,
        test_predictions,
    )

    logger.info(
        "Final test metrics: %s",
        test_metrics.to_dict(),
    )

    logger.info("Model training workflow completed")

    return TrainingWorkflowResult(
        selected_model_name=selected.name,
        selected_model=selected.model,
        validation_metrics=selected.metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
    )
