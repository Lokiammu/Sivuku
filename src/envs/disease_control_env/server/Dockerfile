ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

COPY src/core/ /app/src/core/
COPY src/envs/disease_control_env/ /app/src/envs/disease_control_env/

ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "envs.disease_control_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

