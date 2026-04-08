# Self-Evolving Trading Agent — OpenEnv HTTP server
# Env is at repo root (flattened structure). openenv validate passes from root.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy the flat env at repo root
COPY pyproject.toml uv.lock ./
COPY models.py tasks.py client.py openenv.yaml ./
COPY server/ ./server/
COPY rubrics/ ./rubrics/
COPY agents/ ./agents/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# ── final stage ─────────────────────────────────────────────────────────────
FROM ${BASE_IMAGE}

# Install curl in final stage for HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV TRADING_TICKER=AAPL
ENV TRADING_INTERVAL=1d
ENV TRADING_PERIOD=5y
ENV TRADING_INITIAL_CASH=10000
ENV TRADING_MAX_STEPS=500
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "/app/.venv/bin/python -m server.app --host 0.0.0.0 --port ${PORT}"]
