
"""Export dataset structure and quality metadata to Excel."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from price_prediction.data.ingestion import IngesterFactory
from price_prediction.observability.logging import configure_logging

logger = logging.getLogger(__name__)


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Describe columns without exporting their actual records."""

    profile: list[dict] = []

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        non_null = series.dropna()
        dtype = str(series.dtype)

        row = {
            "column_name": column,
            "data_type": dtype,
            "inferred_role": (
                "numeric"
                if pd.api.types.is_numeric_dtype(series)
                else "datetime"
                if pd.api.types.is_datetime64_any_dtype(series)
                else "categorical_or_text"
            ),
            "non_null_count": int(series.notna().sum()),
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_count / len(df) * 100, 2
            ) if len(df) else 0.0,
            "unique_values": int(series.nunique()),
            "is_constant": bool(series.nunique() <= 1),
            "memory_usage_bytes": int(
                series.memory_usage(index=False, deep=True)
            ),
            "minimum": None,
            "maximum": None,
            "mean": None,
            "std_dev": None,
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(
                series, errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if not numeric.empty:
                row.update({
                    "minimum": numeric.min(),
                    "maximum": numeric.max(),
                    "mean": numeric.mean(),
                    "std_dev": numeric.std(),
                })

        profile.append(row)

    return pd.DataFrame(profile)


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create dataset-level metadata."""

    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())

    summary = {
        "metric": [
            "row_count",
            "column_count",
            "duplicate_rows",
            "missing_cells",
            "missing_cells_percentage",
            "memory_usage_bytes",
        ],
        "value": [
            len(df),
            len(df.columns),
            int(df.duplicated().sum()),
            missing_cells,
            round(missing_cells / total_cells * 100, 2)
            if total_cells else 0.0,
            int(df.memory_usage(deep=True).sum()),
        ],
    }

    return pd.DataFrame(summary)


def build_quality_indicators(
    profile: pd.DataFrame,
    high_missing_threshold: float = 40.0,
) -> pd.DataFrame:
    """Flag columns that may need investigation."""

    indicators = []

    for _, column in profile.iterrows():
        issues = []

        if column["missing_percentage"] > high_missing_threshold:
            issues.append("High missingness")

        if column["is_constant"]:
            issues.append("Constant or entirely missing")

        if issues:
            indicators.append({
                "column_name": column["column_name"],
                "data_type": column["data_type"],
                "missing_percentage": column["missing_percentage"],
                "unique_values": column["unique_values"],
                "potential_issues": "; ".join(issues),
            })

    return pd.DataFrame(
        indicators,
        columns=[
            "column_name",
            "data_type",
            "missing_percentage",
            "unique_values",
            "potential_issues",
        ],
    )


def export_profile(source: str, output: Path) -> None:
    """Ingest a dataset and export structural metadata."""

    df = IngesterFactory.get(source).ingest(source)

    if df.empty:
        raise ValueError("Cannot profile an empty dataset.")

    profile = build_column_profile(df)
    summary = build_dataset_summary(df)
    indicators = build_quality_indicators(profile)

    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        profile.to_excel(
            writer, sheet_name="Column Profile", index=False
        )
        summary.to_excel(
            writer, sheet_name="Dataset Summary", index=False
        )
        indicators.to_excel(
            writer, sheet_name="Quality Indicators", index=False
        )

        # Make the workbook easier to inspect.
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for cells in worksheet.columns:
                column_letter = cells[0].column_letter
                max_length = max(
                    len(str(cell.value or "")) for cell in cells
                )
                worksheet.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 12), 35
                )

    logger.info("Dataset profile exported: %s", output.resolve())
    logger.info(
        "Profile summary: rows=%d, columns=%d, missing_cells=%d",
        len(df),
        len(df.columns),
        int(df.isna().sum().sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export dataset structure to Excel without exporting records."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local CSV path or s3://bucket/key URI.",
    )
    parser.add_argument(
        "--output",
        default="reports/dataset_profile.xlsx",
        help="Output Excel path.",
    )
    args = parser.parse_args()

    configure_logging()
    export_profile(args.source, Path(args.output))


if __name__ == "__main__":
    main()
