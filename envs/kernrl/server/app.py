# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the kernrl GPU kernel optimization environment.

Usage:
    # Development:
    uvicorn kernrl.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn kernrl.server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server import create_app

from kernrl.models import KernelAction, KernelObservation
from kernrl.server.kernel_env import KernelOptEnv

# Create the app with OpenEnv's standard interface
app = create_app(KernelOptEnv, KernelAction, KernelObservation, env_name="kernrl")


def main():
    """Main entry point for running the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
