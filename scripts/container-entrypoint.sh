#!/bin/sh
set -eu
ARTIFACT_PATH="${MODEL_ARTIFACT_PATH:-/opt/ml/model/house_price_model.joblib}"
if [ -n "${MODEL_ARTIFACT_S3_URI:-}" ]; then
  export MODEL_ARTIFACT_PATH="$ARTIFACT_PATH"
  python - <<'PY'
import os
from urllib.parse import urlparse
import boto3
uri = os.environ["MODEL_ARTIFACT_S3_URI"]
parsed = urlparse(uri)
if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
    raise SystemExit("MODEL_ARTIFACT_S3_URI must be s3://bucket/key")
path = os.environ["MODEL_ARTIFACT_PATH"]
os.makedirs(os.path.dirname(path), exist_ok=True)
boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), path)
PY
fi
export MODEL_ARTIFACT_PATH="$ARTIFACT_PATH"
exec uvicorn price_prediction.api.main:app --host 0.0.0.0 --port 8000
