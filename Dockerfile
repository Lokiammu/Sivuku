# Self-Evolving Trading Agent — OpenEnv HTTP server
# Uses local src/openenv/ to avoid heavy gradio/torch dependency chain
# that comes with pip installing openenv-core from PyPI.
FROM python:3.11-slim

WORKDIR /app

# System deps: curl only (for HEALTHCHECK) — no build-essential needed, all wheels are pre-built
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install only the packages our server actually needs at runtime.
# --prefer-binary: always pick pre-built wheels, never compile from source.
# Deliberately excludes: gradio, torch, anthropic, typer, rich (unused in server path).
RUN pip install --no-cache-dir --prefer-binary \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "pydantic>=2.0" \
    "websockets>=13.0" \
    "httpx>=0.24" \
    "yfinance>=0.2.40" \
    "pandas>=2.0" \
    "numpy>=1.24" \
    "openai>=1.0" \
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
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "python -m server.app --host 0.0.0.0 --port ${PORT}"]
