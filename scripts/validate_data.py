
"""Validate a dataset without starting model training."""

import argparse
import logging

from price_prediction.data.ingestion import IngesterFactory
from price_prediction.data.validation import validate_schema
from price_prediction.data.quality import run_quality_checks
from price_prediction.domain.schemas import FeatureSchema
from price_prediction.observability.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest and validate a housing dataset."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local CSV path or s3://bucket/key URI.",
    )
    args = parser.parse_args()

    configure_logging()

    logger.info("Starting dataset validation: source=%s", args.source)

    # 1. Ingest dataset
    ingester = IngesterFactory.get(args.source)
    df = ingester.ingest(args.source)

    # 2. Validate raw dataset schema
    df = validate_schema(df)

    # 3. Apply quality checks to model features
    feature_schema = FeatureSchema()

    report = run_quality_checks(
        df,
        target=feature_schema.target,
        feature_columns=feature_schema.all_features,
    )

    # 4. Report results
    logger.info("Dataset validation completed successfully")
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Quality report: %s", report.to_dict())


if __name__ == "__main__":
    main()
