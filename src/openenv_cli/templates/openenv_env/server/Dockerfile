# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# Build from repo root: docker build -t __ENV_NAME___env:latest -f src/envs/__ENV_NAME___env/server/Dockerfile .
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Copy core module (needed by __ENV_NAME___env)
COPY src/core /app/src/core

# Copy environment
COPY src/envs/__ENV_NAME___env /app/src/envs/__ENV_NAME___env

# Install dependencies using uv sync
WORKDIR /app/src/envs/__ENV_NAME___env
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-editable

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy core module
COPY --from=builder /app/src/core /app/src/core

# Copy the virtual environment from builder
COPY --from=builder /app/src/envs/__ENV_NAME___env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/src/envs/__ENV_NAME___env /app/src/envs/__ENV_NAME___env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.__ENV_NAME___env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
