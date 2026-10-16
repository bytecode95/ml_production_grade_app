
"""Fail-fast schema validation for ingested datasets."""

import logging

import pandas as pd
import pandera.pandas as pa

from price_prediction.data.schemas import (
    AmesHousingSchema,
    REQUIRED_COLUMNS,
)
from price_prediction.domain.exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_schema(
    df: pd.DataFrame,
    *,
    required_columns: tuple[str, ...] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """Validate required columns and return the validated DataFrame."""

    if df.empty:
        raise ValidationError("Dataset must contain at least one row.")

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    try:
        validated_df = AmesHousingSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases.head(20).to_dict(
            orient="records"
        )
        logger.error("Schema validation failed: %s", failure_cases)
        raise ValidationError(
            f"Dataset schema validation failed: {failure_cases}"
        ) from exc

    logger.info(
        "Schema validation passed: rows=%d, columns=%d",
        len(validated_df),
        len(validated_df.columns),
    )

    return validated_df
