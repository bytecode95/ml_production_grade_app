FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock* ./
COPY src ./src
COPY scripts/container-entrypoint.sh /usr/local/bin/container-entrypoint
RUN pip install --no-cache-dir --upgrade pip uv \
    && uv pip install --system . \
    && chmod +x /usr/local/bin/container-entrypoint \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /opt/ml/model \
    && chown -R appuser:appuser /app /opt/ml/model
USER appuser
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/container-entrypoint"]
