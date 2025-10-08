# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# EnvTorch Base Image
#
# This is the standard base image for all EnvTorch environment servers.
# It includes the minimal dependencies needed to run HTTP environment servers.
#
# Build: docker build -t envtorch-base:latest -f src/core/docker/base/Dockerfile .
# Tag:   docker tag envtorch-base:latest envtorch-base:0.1.0
#

FROM python:3.11-slim

# Set metadata
LABEL maintainer="EnvTorch Team"
LABEL description="Base image for EnvTorch environment servers"
LABEL version="0.1.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies that all environments need
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    requests>=2.25.0

# Set working directory
WORKDIR /app

# Default environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default expose port (can be overridden)
EXPOSE 8000

# Note: CMD should be specified in child Dockerfiles
