
import logging

import pandas as pd

from price_prediction.data.ingestion import IngesterFactory

logger = logging.getLogger(__name__)


def test_ingest_local_ames_housing_csv() -> None:
    source = "data/AmesHousing.csv"

    logger.info("Starting local CSV ingestion: source=%s", source)

    ingester = IngesterFactory.get(source)
    df = ingester.ingest(source)

    # Verify ingestion
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df.columns) > 0

    # Log dataset diagnostics
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Dataset columns: %s", df.columns.tolist())
    logger.info("Dataset data types:\n%s", df.dtypes)
    logger.info("Dataset sample (first 5 rows):\n%s", df.head())
    logger.info("Missing values by column:\n%s", df.isna().sum())
