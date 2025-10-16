# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build for OpenSpiel + OpenEnv
# Stage 1: Build OpenSpiel using their official Dockerfile
# Using Ubuntu 22.04 for Python 3.10 (closer to modern Python)
FROM ubuntu:22.04 AS openspiel-builder

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies including tzdata
RUN apt update && apt-get -y install \
    tzdata \
    clang \
    curl \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set up OpenSpiel build directory
RUN mkdir /repo
WORKDIR /repo

# Upgrade pip
RUN pip3 install --upgrade pip

# Clone OpenSpiel (we'll use a specific commit or tag for reproducibility)
RUN git clone https://github.com/google-deepmind/open_spiel.git .

# Run OpenSpiel's installation script (handles all C++ dependencies)
RUN ./install.sh

# Install Python dependencies
RUN pip3 install --upgrade setuptools testresources
RUN pip3 install importlib_metadata --force-reinstall
RUN pip3 install --upgrade -r requirements.txt
RUN pip3 install --upgrade cmake

# Build OpenSpiel
RUN mkdir -p build
WORKDIR /repo/build
RUN cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=$(which clang++) ../open_spiel
RUN make -j$(nproc) pyspiel

# Stage 2: Runtime image with OpenSpiel + OpenEnv
# Must use Python 3.10 to match Ubuntu 22.04's Python version in builder
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy OpenSpiel build artifacts from builder
RUN mkdir -p /repo
WORKDIR /repo
COPY --from=openspiel-builder /repo /repo

# Install OpenSpiel Python requirements in runtime
RUN pip3 install --upgrade -r requirements.txt

# Set Python path for OpenSpiel
ENV PYTHONPATH=/repo:/repo/build/python:${PYTHONPATH}

# Create app directory for OpenEnv
WORKDIR /app

# Install OpenEnv dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    requests>=2.25.0

# Copy OpenEnv core
COPY src/core/ /app/src/core/

# Copy OpenSpiel environment
COPY src/envs/openspiel_env/ /app/src/envs/openspiel_env/

# Set Python path for OpenEnv (needs /app/src so 'envs' module is found)
ENV PYTHONPATH=/app/src:${PYTHONPATH}

# Default environment variables (can be overridden)
ENV OPENSPIEL_GAME=catch
ENV OPENSPIEL_AGENT_PLAYER=0
ENV OPENSPIEL_OPPONENT_POLICY=random

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["python3", "-m", "uvicorn", "envs.openspiel_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
