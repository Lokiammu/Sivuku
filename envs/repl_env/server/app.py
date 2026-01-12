# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the REPL Environment.

This module creates an HTTP server that exposes the REPLEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

The server includes llm_query and llm_batch support via HuggingFace Inference API,
enabling the Recursive Language Model (RLM) paradigm.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server

Environment Variables:
    HF_TOKEN: HuggingFace API token for Inference API (optional for public models)
    LLM_MODEL: Model to use for llm_query/llm_batch (default: Qwen/Qwen3-1.7B)
"""
import os
from typing import List

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app
    from ..models import REPLAction, REPLObservation
    from .repl_environment import REPLEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app
    from models import REPLAction, REPLObservation
    from server.repl_environment import REPLEnvironment


# ============== LLM CONFIGURATION ==============
# Model to use for llm_query and llm_batch
# Using Qwen2.5 (no thinking mode, simpler responses)
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
# ===============================================


def create_llm_query_fn():
    """
    Create the llm_query function using HuggingFace Inference API.
    
    Returns:
        Function that takes a prompt string and returns the model's response.
    """
    from huggingface_hub import InferenceClient
    
    client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
    
    def llm_query(prompt: str) -> str:
        """Query the LLM with a prompt and return the response."""
        try:
            # Use chat_completion for chat models like Qwen
            messages = [{"role": "user", "content": prompt}]
            response = client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {e}"
    
    return llm_query


def create_llm_batch_fn(llm_query_fn):
    """
    Create the llm_batch function for processing multiple prompts.
    
    Args:
        llm_query_fn: The single-query LLM function
        
    Returns:
        Function that takes a list of prompts and returns a list of responses.
    """
    def llm_batch(prompts: List[str]) -> List[str]:
        """Query the LLM with multiple prompts and return all responses."""
        # For now, process sequentially. Could be parallelized with async.
        return [llm_query_fn(prompt) for prompt in prompts]
    
    return llm_batch


def create_repl_environment_factory():
    """
    Factory function that creates REPLEnvironment instances with LLM support.
    
    This factory is called for each new WebSocket session, creating a fresh
    environment with llm_query and llm_batch functions enabled.
    """
    # Create LLM functions (shared across instances for efficiency)
    llm_query_fn = create_llm_query_fn()
    llm_batch_fn = create_llm_batch_fn(llm_query_fn)
    
    def factory():
        return REPLEnvironment(
            llm_query_fn=llm_query_fn,
            llm_batch_fn=llm_batch_fn,
        )
    
    return factory


# Check if LLM support should be enabled
ENABLE_LLM = os.environ.get("ENABLE_LLM", "true").lower() in ("true", "1", "yes")

if ENABLE_LLM:
    print(f"[REPL Server] LLM support ENABLED with model: {LLM_MODEL}")
    env_factory = create_repl_environment_factory()
else:
    print("[REPL Server] LLM support DISABLED")
    env_factory = REPLEnvironment

# Create the app with web interface and README integration
app = create_app(env_factory, REPLAction, REPLObservation, env_name="repl_env")


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.repl_env.server.app
        openenv serve repl_env
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
