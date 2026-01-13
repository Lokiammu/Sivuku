# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Julia Environment.

This module creates an HTTP server that exposes the JuliaCodeActEnv
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.julia_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.julia_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.julia_env.server.app
"""

from openenv.core.env_server.http_server import create_app

from julia_env.models import JuliaAction, JuliaObservation
from julia_env.server.julia_codeact_env import JuliaCodeActEnv

# Create the app with web interface and README integration
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(JuliaCodeActEnv, JuliaAction, JuliaObservation, env_name="julia_env")


def main():
    """Main entry point for running the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
