"""Evaluate SageMaker model artifact against held-out test data."""
import json
import tarfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    model_root = Path("/opt/ml/processing/model")
    archives = list(model_root.glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError("Expected model.tar.gz input.")
    extracted = model_root / "unpacked"
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archives[0], "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    candidates = list(extracted.rglob("model.joblib"))
    if not candidates:
        raise FileNotFoundError("model.joblib missing from artifact.")
    model = joblib.load(candidates[0])
    test_files = list(Path("/opt/ml/processing/test").glob("*.csv"))
    if not test_files:
        raise FileNotFoundError("Test CSV missing.")
    df = pd.read_csv(test_files[0])
    if "SalePrice" not in df:
        raise ValueError("Test CSV must contain SalePrice.")
    y = df.pop("SalePrice").to_numpy(dtype=float)
    predictions = np.asarray(model.predict(df), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")
    report = {
        "mae": float(mean_absolute_error(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "r2": float(r2_score(y, predictions)),
    }
    output = Path("/opt/ml/processing/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
