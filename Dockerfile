# Self-Evolving Trading Agent — OpenEnv HTTP server
# Flat structure — openenv validate passes from repo root.
FROM python:3.11-slim

WORKDIR /app

# Install system deps + curl for HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly with pip (no uv/lockfile needed)
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.3" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "yfinance>=0.2.40" \
    "pandas>=2.0" \
    "numpy>=1.24" \
    "pydantic>=2.0" \
    "openai>=1.0"

# Copy application code
COPY models.py tasks.py client.py openenv.yaml ./
COPY server/ ./server/
COPY rubrics/ ./rubrics/
COPY agents/ ./agents/

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

CMD ["sh", "-c", "python -m server.app --host 0.0.0.0 --port ${PORT}"]
