
"""Pandera schemas for raw housing datasets."""

"""
schemas.py — define the expected dataset structure, column types, and target column.
validation.py — enforce the schema and fail fast when required columns or data types are invalid.
quality.py — check missing values, duplicate rows, invalid values, and configurable quality thresholds.

"""

import pandera.pandas as pa
from pandera.typing import Series

"""
pa provides DataFrameModel and Field to define validation rules.
Series declares the expected type of each DataFrame column.

"""

"""
A Pandera schema is a set of rules that defines what your Pandas DataFrame should look like and what data it is allowed to contain.
Definition	What it enforces
SalePrice: Series[float]	Target should be numeric.
gt=0	                    Value must be greater than zero.
nullable=False	            Null values are not allowed.
coerce=True	                Pandera attempts type conversion before validating.
ge=1, le=10	                Overall quality must be between 1 and 10.
alias="Gr Liv Area"	         Maps the Python attribute name to the actual dataset column name.

"""

class AmesHousingSchema(pa.DataFrameModel):
    """Validate required raw training columns and constraints."""

    SalePrice: Series[float] = pa.Field(
        gt=0, nullable=False, coerce=True
    )

    Gr_Liv_Area: Series[float] = pa.Field(
        alias="Gr Liv Area", gt=0, nullable=False, coerce=True
    )

    Overall_Qual: Series[int] = pa.Field(
        alias="Overall Qual",
        ge=1,
        le=10,
        nullable=False,
        coerce=True,
    )

    Year_Built: Series[int] = pa.Field(
        alias="Year Built", ge=1870, nullable=False, coerce=True
    )


DEFAULT_TARGET_COLUMN = "SalePrice"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "SalePrice",
    "Gr Liv Area",
    "Total Bsmt SF",
    "Garage Cars",
    "Overall Qual",
    "Year Built",
    "Full Bath",
    "Bedroom AbvGr",
    "Lot Area",
    "TotRms AbvGrd",
    "Fireplaces",
    "Neighborhood",
    "House Style",
    "Sale Condition",
)
