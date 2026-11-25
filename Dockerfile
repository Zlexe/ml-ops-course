FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_VERSION=v1.0.0

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 appuser

COPY --from=ghcr.io/astral-sh/uv:0.13.10 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
ENV UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=10
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY models ./models

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["/app/.venv/bin/uvicorn", "ml_ops_course.main:app", "--host", "0.0.0.0", "--port", "8000"]
