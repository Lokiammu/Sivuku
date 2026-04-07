# Submission root Dockerfile — builds the trading env FastAPI server.
# For the HuggingFace Space dashboard see dashboard/Dockerfile (auto-detected
# by Spaces via dashboard/README.md front-matter).
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

COPY envs/trading_env /app/env
COPY rubrics /app/rubrics
WORKDIR /app/env

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

FROM ${BASE_IMAGE}

WORKDIR /app
COPY --from=builder /app /app

ENV PYTHONPATH="/app/env:/app:$PYTHONPATH"
ENV TRADING_TICKER=AAPL
ENV TRADING_INTERVAL=1d
ENV TRADING_PERIOD=5y
ENV TRADING_INITIAL_CASH=10000
ENV TRADING_MAX_STEPS=500
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && .venv/bin/python -m server.app --host 0.0.0.0 --port ${PORT}"]
