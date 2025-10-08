# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Docker utilities for running environment servers locally.

This module provides low-level Docker helpers for container management,
used by HTTPEnvClient.from_docker_image() and subclasses.
"""

from __future__ import annotations

from typing import Any


def _find_available_port() -> int:
    """
    Find an available port on localhost.

    Returns:
        An available port number
    """
    # TODO: Implement port finding logic
    # Can use socket to find an available port
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _generate_container_name(image: str) -> str:
    """
    Generate a unique container name based on image name and timestamp.

    Args:
        image: Docker image name

    Returns:
        A unique container name
    """
    # TODO: Implement container name generation
    # Extract image name without tag/registry
    # Add timestamp or UUID for uniqueness
    import time

    clean_image = image.split("/")[-1].split(":")[0]
    timestamp = int(time.time() * 1000)
    return f"{clean_image}-{timestamp}"


def _start_container(image: str, name: str, port: int) -> Any:
    """
    Start a Docker container from the specified image.

    Args:
        image: Docker image name
        name: Container name
        port: Port to expose

    Returns:
        Container handle/reference for cleanup
    """
    # TODO: Implement using docker SDK or subprocess
    # docker_client = docker.from_env()
    # container = docker_client.containers.run(
    #     image,
    #     name=name,
    #     ports={"8000/tcp": port},  # Assuming server runs on 8000 internally
    #     detach=True,
    #     remove=True,  # Auto-remove when stopped
    # )
    # return container
    raise NotImplementedError("Docker container start logic not yet implemented")


def _wait_for_server_ready(base_url: str) -> None:
    """
    Wait for the environment server to be ready to accept requests.

    Args:
        base_url: Base URL of the server

    Raises:
        TimeoutError: If server doesn't become ready within timeout
    """
    # TODO: Implement health check polling
    # import time
    # import requests
    #
    # timeout_s = 30.0
    # start_time = time.time()
    # while time.time() - start_time < timeout_s:
    #     try:
    #         response = requests.get(f"{base_url}/health", timeout=1.0)
    #         if response.status_code == 200:
    #             return
    #     except requests.RequestException:
    #         pass
    #     time.sleep(0.5)
    # raise TimeoutError(f"Server at {base_url} did not become ready within {timeout_s}s")
    raise NotImplementedError("Server health check logic not yet implemented")


def _stop_container(container: Any) -> None:
    """
    Stop and remove a Docker container.

    Args:
        container: Container handle from _start_container
    """
    # TODO: Implement container cleanup
    # try:
    #     container.stop(timeout=5)
    #     container.remove()
    # except Exception as e:
    #     # Log error but don't fail
    #     print(f"Warning: Failed to clean up container: {e}")
    raise NotImplementedError("Docker container cleanup logic not yet implemented")
