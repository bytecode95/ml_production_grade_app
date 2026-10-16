"""Prediction API schemas."""

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

PositiveNumber = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeNumber = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]


class HouseFeatures(BaseModel):
    """Validated input features for one house."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    gr_liv_area: PositiveNumber = Field(alias="Gr Liv Area")
    total_bsmt_sf: NonNegativeNumber | None = Field(
        default=None,
        alias="Total Bsmt SF",
    )
    garage_cars: NonNegativeNumber | None = Field(
        default=None,
        alias="Garage Cars",
    )
    overall_qual: Annotated[int, Field(ge=1, le=10)] = Field(
        alias="Overall Qual"
    )
    year_built: Annotated[int, Field(ge=1800, le=2100)] = Field(
        alias="Year Built"
    )
    full_bath: Annotated[int, Field(ge=0, le=20)] = Field(
        alias="Full Bath"
    )
    bedroom_abv_gr: Annotated[int, Field(ge=0, le=50)] = Field(
        alias="Bedroom AbvGr"
    )
    lot_area: PositiveNumber = Field(alias="Lot Area")
    tot_rms_abv_grd: Annotated[int, Field(ge=0, le=100)] = Field(
        alias="TotRms AbvGrd"
    )
    fireplaces: Annotated[int, Field(ge=0, le=20)] = Field(
        alias="Fireplaces"
    )
    neighborhood: NonEmptyString = Field(alias="Neighborhood")
    house_style: NonEmptyString = Field(alias="House Style")
    sale_condition: NonEmptyString = Field(alias="Sale Condition")


class PredictionRequest(BaseModel):
    """Request containing one or more houses."""

    model_config = ConfigDict(extra="forbid")

    houses: Annotated[
        list[HouseFeatures],
        Field(min_length=1, max_length=100),
    ]


class HousePrediction(BaseModel):
    """Prediction for one house."""

    predicted_price: float


class PredictionResponse(BaseModel):
    """Stable response contract for prediction requests."""

    predictions: list[HousePrediction]
    model_name: str
    model_version: str
    target: str


class ModelInfoResponse(BaseModel):
    """Safe model metadata exposed through the API."""

    model_name: str
    model_version: str
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]
    target_transform: str
    created_at: str