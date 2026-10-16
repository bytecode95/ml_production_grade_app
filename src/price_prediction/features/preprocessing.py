
"""Reusable numeric and categorical preprocessing."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from price_prediction.domain.schemas import FeatureSchema


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    """Build an unfitted preprocessor from the canonical feature schema."""

    numeric_features = schema.numeric_features
    categorical_features = schema.categorical_features

    if not numeric_features and not categorical_features:
        raise ValueError("At least one model feature must be configured.")

    overlap = set(numeric_features) & set(categorical_features)
    if overlap:
        raise ValueError(
            f"Features cannot be both numeric and categorical: "
            f"{sorted(overlap)}"
        )

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            ("num", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("cat", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
