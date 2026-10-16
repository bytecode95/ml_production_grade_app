"""Deterministic SageMaker data split."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    source = Path("/opt/ml/processing/input")
    csvs = list(source.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {source}")
    df = pd.read_csv(csvs[0])
    if df.empty or "SalePrice" not in df:
        raise ValueError("Expected non-empty CSV with SalePrice target.")
    if df["SalePrice"].isna().any():
        raise ValueError("Target SalePrice contains missing values.")
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, validation = train_test_split(
        train_val, test_size=0.15 / 0.85, random_state=42
    )
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        folder = Path("/opt/ml/processing") / name
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
