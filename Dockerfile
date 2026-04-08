# syntax=docker/dockerfile:1.4
# Self-Evolving Trading Agent — OpenEnv HTTP server
# Uses local src/openenv/ to avoid heavy gradio/torch dependency chain
# that comes with pip installing openenv-core from PyPI.
FROM python:3.11-slim

WORKDIR /app

# System deps: curl only (for HEALTHCHECK) — no build-essential, all wheels are pre-built
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install only the packages the server actually needs at runtime.
# --prefer-binary: always pick pre-built wheels, never compile from source.
# BuildKit cache mount keeps the pip download cache across builds so we don't
# re-download pandas/numpy/etc. every time a line in the Dockerfile changes.
#
# Excluded (unused in server path):
#   gradio, torch, anthropic, openai, typer, rich, fastmcp
# Also excluded:
#   yfinance — market_sim.py imports it inside a try/except and falls back
#   to synthetic GBM data when missing. Tasks use deterministic scenarios
#   so live market data is never needed on HF Spaces. Dropping yfinance
#   also drops cryptography, lxml, beautifulsoup4, requests, html5lib,
#   peewee — saves ~30s of pip install time and ~40MB of downloads.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "pydantic>=2.0" \
    "websockets>=13.0" \
    "httpx>=0.24" \
    "pandas>=2.0" \
    "numpy>=1.24" \
    "pyyaml"

# Copy patched local openenv — avoids full openenv-core PyPI package
# (which pulls gradio + heavy ML deps we don't need)
COPY src/ ./src/

# Copy application code
COPY models.py tasks.py client.py openenv.yaml ./
COPY server/ ./server/
COPY rubrics/ ./rubrics/
COPY agents/ ./agents/

# src/ first so our patched openenv (with task_score serialization fix)
# takes precedence over any cached/conflicting installs
ENV PYTHONPATH="/app/src:/app:$PYTHONPATH"
ENV TRADING_TICKER=AAPL
ENV TRADING_INTERVAL=1d
ENV TRADING_PERIOD=5y
ENV TRADING_INITIAL_CASH=10000
ENV TRADING_MAX_STEPS=500
# HF Spaces Docker SDK defaults to probing port 7860 — bind there so the
# Space leaves "Starting" state. Override with PORT env var if running elsewhere.
ENV PORT=7860

EXPOSE 7860

# Tight interval so HF flips "Running" quickly (first check at 5s, not 30s).
HEALTHCHECK --interval=5s --timeout=3s --start-period=3s --retries=5 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "python -m server.app --host 0.0.0.0 --port ${PORT}"]
