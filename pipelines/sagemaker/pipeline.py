
"""SageMaker ML pipeline for housing-price prediction.

Pipeline:
    PrepareData -> TrainModel -> EvaluateModel
                                  |
                                  v
                             ModelQualityGate
                                  |
                                  v
                          RegisterCandidate

Requires SageMaker Python SDK v2.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


ROOT = Path(__file__).resolve().parents[2]


def build_pipeline(
    *,
    name: str,
    role: str,
    bucket: str,
    region: str,
) -> Pipeline:
    """Construct the SageMaker processing/training/evaluation pipeline."""

    session = Session(default_bucket=bucket)

    # Runtime parameters make the pipeline reusable across datasets.
    training_data_uri = ParameterString(
        name="TrainingDataUri",
        default_value=f"s3://{bucket}/datasets/ames_housing.csv",
    )

    minimum_r2 = ParameterFloat(
        name="MinimumR2",
        default_value=0.50,
    )

    approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )

    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.large",
    )

    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.m5.large",
    )

    # Use the same supported sklearn image for the processing scripts.
    sklearn_image = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.large",
    )

    processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=session,
        base_job_name="price-prediction-processing",
    )

    # ---------------------------------------------------------
    # Step 1: Prepare train, validation and test partitions.
    # ---------------------------------------------------------
    prepare_step = ProcessingStep(
        name="PrepareData",
        processor=processor,
        code=str(ROOT / "scripts" / "sagemaker" / "prepare.py"),
        inputs=[
            ProcessingInput(
                input_name="raw-data",
                source=training_data_uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
            ),
        ],
    )

    train_s3_uri = (
        prepare_step.properties
        .ProcessingOutputConfig.Outputs["train"]
        .S3Output.S3Uri
    )

    test_s3_uri = (
        prepare_step.properties
        .ProcessingOutputConfig.Outputs["test"]
        .S3Output.S3Uri
    )

    # ---------------------------------------------------------
    # Step 2: Train the existing project pipeline.
    #
    # source_dir=ROOT includes the src/ package as well as
    # scripts/sagemaker/train.py in the submitted source bundle.
    # ---------------------------------------------------------
    estimator = SKLearn(
        entry_point="scripts/sagemaker/train.py",
        source_dir=str(ROOT),
        framework_version="1.2-1",
        py_version="py3",
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        output_path=f"s3://{bucket}/price-prediction/training",
        sagemaker_session=session,
        base_job_name="price-prediction-training",
        hyperparameters={
            "target-column": "SalePrice",
            "model-name": "random_forest",
            "random-state": 42,
            "n-estimators": 200,
            "max-depth": 15,
        },
    )

    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=train_s3_uri,
                content_type="text/csv",
            ),
        },
    )

    model_artifact_uri = train_step.properties.ModelArtifacts.S3ModelArtifacts

    # ---------------------------------------------------------
    # Step 3: Evaluate on the untouched test partition.
    # ---------------------------------------------------------
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluate_step = ProcessingStep(
        name="EvaluateModel",
        processor=processor,
        code=str(ROOT / "scripts" / "sagemaker" / "evaluate.py"),
        inputs=[
            ProcessingInput(
                input_name="model-artifact",
                source=model_artifact_uri,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                input_name="test-data",
                source=test_s3_uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            ),
        ],
        property_files=[evaluation_report],
    )

    evaluation_s3_uri = (
        evaluate_step.properties
        .ProcessingOutputConfig.Outputs["evaluation"]
        .S3Output.S3Uri
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[evaluation_s3_uri, "evaluation.json"],
            ),
            content_type="application/json",
        ),
    )

    # ---------------------------------------------------------
    # Step 4: Register only when the model meets the R2 gate.
    # Registration does not automatically deploy the model.
    # ---------------------------------------------------------
    register_step = RegisterModel(
        name="RegisterCandidate",
        estimator=estimator,
        model_data=model_artifact_uri,
        content_types=["text/csv", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"{name}-models",
        approval_status=approval_status,
        model_metrics=model_metrics,
    )

    r2_value = (
        evaluate_step.properties
        .PropertyFiles["EvaluationReport"]
        .Outputs["r2"]
    )

    quality_gate = ConditionStep(
        name="ModelQualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=r2_value,
                right=minimum_r2,
            ),
        ],
        if_steps=[register_step],
        else_steps=[],
    )

    return Pipeline(
        name=name,
        parameters=[
            training_data_uri,
            minimum_r2,
            approval_status,
            processing_instance_type,
            training_instance_type,
        ],
        steps=[
            prepare_step,
            train_step,
            evaluate_step,
            quality_gate,
        ],
        sagemaker_session=session,
    )


def main() -> None:
    """Upsert the pipeline and optionally start an execution."""

    parser = argparse.ArgumentParser(
        description="Manage the housing-price SageMaker pipeline."
    )

    parser.add_argument(
        "--start",
        action="store_true",
        help="Start an execution after upserting the definition.",
    )

    parser.add_argument(
        "--training-data-uri",
        help="S3 URI of the source CSV.",
    )

    parser.add_argument(
        "--minimum-r2",
        type=float,
        default=float(os.getenv("MINIMUM_R2", "0.50")),
    )

    args = parser.parse_args()

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    role = os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN")
    bucket = os.getenv("SAGEMAKER_PIPELINE_BUCKET")
    name = os.getenv(
        "SAGEMAKER_PIPELINE_NAME",
        "price-prediction-training",
    )

    if not region:
        raise RuntimeError("AWS_REGION must be configured.")

    if not role:
        raise RuntimeError(
            "SAGEMAKER_EXECUTION_ROLE_ARN must be configured."
        )

    if not bucket:
        raise RuntimeError(
            "SAGEMAKER_PIPELINE_BUCKET must be configured."
        )

    if not 0.0 <= args.minimum_r2 <= 1.0:
        raise ValueError(
            "--minimum-r2 must be between 0 and 1."
        )

    pipeline = build_pipeline(
        name=name,
        role=role,
        bucket=bucket,
        region=region,
    )

    # Creates or updates the pipeline definition in AWS.
    pipeline.upsert(role_arn=role)
    print(f"Pipeline definition updated: {name}")

    if not args.start:
        print("No execution started. Use --start to run training.")
        return

    data_uri = (
        args.training_data_uri
        or os.getenv("TRAINING_DATA_URI")
    )

    if not data_uri or not data_uri.startswith("s3://"):
        raise ValueError(
            "Provide a valid S3 URI using --training-data-uri "
            "or TRAINING_DATA_URI."
        )

    execution = pipeline.start(
        parameters={
            "TrainingDataUri": data_uri,
            "MinimumR2": args.minimum_r2,
            "ModelApprovalStatus": "PendingManualApproval",
        }
    )

    print(f"Pipeline execution started: {execution.arn}")


if __name__ == "__main__":
    main()
