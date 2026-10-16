"""Core domain contracts. Pure pydantic — no frameworks, no IO."""
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


class FeatureSchema(BaseModel):
    """Canonical feature contract shared by training and inference."""
    target: str = "SalePrice"
    numeric_features: list[str] = Field(default_factory=lambda: [
        "Gr Liv Area", "Total Bsmt SF", "Garage Cars", "Overall Qual",
        "Year Built", "Full Bath", "Bedroom AbvGr", "Lot Area",
        "TotRms AbvGrd", "Fireplaces",
    ])
    categorical_features: list[str] = Field(default_factory=lambda: [
        "Neighborhood", "House Style", "Sale Condition",
    ])

    @property
    def all_features(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    def to_frame(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        missing = set(self.all_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        return df[self.all_features]


class RawInput(BaseModel):
    """Validated API input for a single prediction."""
    Gr_Liv_Area: float = Field(..., alias="Gr Liv Area")
    Total_Bsmt_SF: float = Field(..., alias="Total Bsmt SF")
    Garage_Cars: float
    Overall_Qual: float = Field(..., ge=1, le=10)
    Year_Built: int = Field(..., ge=1870, le=2026)
    Full_Bath: float
    Bedroom_AbvGr: float
    Lot_Area: float
    TotRms_AbvGrd: float
    Fireplaces: float
    Neighborhood: str
    House_Style: str
    Sale_Condition: str

    model_config = {"populate_by_name": True}

    def to_feature_row(self) -> dict:
        return {
            "Gr Liv Area": self.Gr_Liv_Area,
            "Total Bsmt SF": self.Total_Bsmt_SF,
            "Garage Cars": self.Garage_Cars,
            "Overall Qual": self.Overall_Qual,
            "Year Built": self.Year_Built,
            "Full Bath": self.Full_Bath,
            "Bedroom AbvGr": self.Bedroom_AbvGr,
            "Lot Area": self.Lot_Area,
            "TotRms AbvGrd": self.TotRms_AbvGrd,
            "Fireplaces": self.Fireplaces,
            "Neighborhood": self.Neighborhood,
            "House Style": self.House_Style,
            "Sale Condition": self.Sale_Condition,
        }


class PredictionResult(BaseModel):
    predicted_price: float
    model_version: str
    model_name: str
    request_id: str
    created_at: datetime
    status: str = "success"


class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    r2: float
    mape: float
