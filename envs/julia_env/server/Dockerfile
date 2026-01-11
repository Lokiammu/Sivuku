# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app/env

# Install system dependencies (including curl for Julia installation)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install juliaup and Julia
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.10

# Add Julia to PATH
ENV PATH="/root/.juliaup/bin:${PATH}"

# Verify Julia installation
RUN julia --version

# Precompile commonly used Julia packages (Test is built-in, but precompile it)
RUN julia -e 'using Test; println("Julia Test module ready")'

# Copy environment files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "julia_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
