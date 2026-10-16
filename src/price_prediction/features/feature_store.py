
"""Local Parquet feature store with version metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class LocalFeatureStore:
    """Persist and retrieve versioned feature DataFrames locally."""

    def __init__(self, root: str | Path = "data/features") -> None:
        self.root = Path(root)

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        version: str,
    ) -> str:
        """Save features and metadata; return the Parquet file path."""

        self._validate_identifier(name, "name")
        self._validate_identifier(version, "version")

        if df.empty:
            raise ValueError("Cannot save an empty feature DataFrame.")

        if not df.columns.is_unique:
            raise ValueError("Feature DataFrame has duplicate columns.")

        output_dir = self.root / name / version
        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = output_dir / "features.parquet"
        manifest_path = output_dir / "manifest.json"

        df.to_parquet(parquet_path, index=False)

        manifest = {
            "name": name,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": [
                {"name": str(column), "dtype": str(dtype)}
                for column, dtype in df.dtypes.items()
            ],
            "file": parquet_path.name,
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        return str(parquet_path)

    def load(
        self,
        name: str,
        version: str,
    ) -> pd.DataFrame:
        """Load a saved feature DataFrame."""

        self._validate_identifier(name, "name")
        self._validate_identifier(version, "version")

        feature_dir = self.root / name / version
        parquet_path = feature_dir / "features.parquet"
        manifest_path = feature_dir / "manifest.json"

        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Feature dataset not found: {parquet_path}"
            )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Feature manifest not found: {manifest_path}"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("name") != name:
            raise ValueError("Feature manifest name does not match.")

        if manifest.get("version") != version:
            raise ValueError("Feature manifest version does not match.")

        df = pd.read_parquet(parquet_path)

        if len(df) != manifest.get("row_count"):
            raise ValueError("Feature row count does not match manifest.")

        expected_columns = manifest.get("columns", [])
        actual_columns = [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in df.dtypes.items()
        ]

        if actual_columns != expected_columns:
            raise ValueError(
                "Loaded feature columns or dtypes do not match manifest."
            )

        return df

    @staticmethod
    def _validate_identifier(value: str, field: str) -> None:
        if (
            not value
            or value in {".", ".."}
            or Path(value).name != value
            or "/" in value
            or "\\" in value
        ):
            raise ValueError(
                f"Invalid feature-store {field}: {value!r}"
            )
