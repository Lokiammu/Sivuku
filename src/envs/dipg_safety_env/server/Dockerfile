# src/envs/dipg_safety_env/server/Dockerfile
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Copy environment code
COPY src/core/ /app/src/core/
COPY src/envs/dipg_safety_env/ /app/src/envs/dipg_safety_env/

# ===> ADD THIS LINE <===
# Copy your dataset so the environment can read it.
# Make sure the path is correct relative to the Docker build context.
COPY harmonic_reasoner_dataset_structured.jsonl /app/harmonic_reasoner_dataset_structured.jsonl

# Health check & CMD (same as example)
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "envs.dipg_safety_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]