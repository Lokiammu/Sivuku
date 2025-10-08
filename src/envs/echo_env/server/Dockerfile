# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the standard envtorch base image
# Built from: docker build -t envtorch-base:latest -f src/core/docker/base/Dockerfile .
FROM envtorch-base:latest

# Copy only what's needed for this environment
COPY src/core/ /app/src/core/
COPY src/envs/echo_env/ /app/src/envs/echo_env/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.echo_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
