
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngester(ABC):
    @abstractmethod
    def ingest(self, source: str) -> pd.DataFrame:
        """Load a dataset from a supported source."""
        raise NotImplementedError


class LocalCSVIngester(DataIngester):
    def ingest(self, source: str) -> pd.DataFrame:
        path = Path(source)

        logger.info("Loading CSV from local path: %s", path)

        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)

        logger.info(
            "Local CSV ingestion completed: rows=%d, columns=%d",
            len(df),
            len(df.columns),
        )
        return df


class S3CSVIngester(DataIngester):
    def __init__(self, region: str = "us-east-1") -> None:
        import boto3

        self._s3 = boto3.client("s3", region_name=region)

    def ingest(self, source: str) -> pd.DataFrame:
        parsed = urlparse(source)

        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"Invalid S3 URI: {source}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        logger.info("Loading CSV from S3: bucket=%s, key=%s", bucket, key)

        response = self._s3.get_object(Bucket=bucket, Key=key)

        with response["Body"] as body:
            content = body.read()

        df = pd.read_csv(BytesIO(content))

        logger.info(
            "S3 CSV ingestion completed: rows=%d, columns=%d",
            len(df),
            len(df.columns),
        )
        return df


class IngesterFactory:
    @staticmethod
    def get(source: str) -> DataIngester:
        source = source.strip()

        if not source:
            raise ValueError("Data source cannot be empty")

        if source.startswith("s3://"):
            return S3CSVIngester()

        if Path(source).suffix.lower() == ".csv":
            return LocalCSVIngester()

        raise ValueError(f"Unsupported data source: {source}")
