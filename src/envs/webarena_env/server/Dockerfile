ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install system dependencies for Playwright and browsers
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Playwright browser dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libxshmfence1 \
    # Additional dependencies
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install WebArena requirements
COPY src/envs/webarena_env/server/requirements.txt /tmp/webarena_requirements.txt
RUN pip install --no-cache-dir -r /tmp/webarena_requirements.txt && \
    rm /tmp/webarena_requirements.txt

# Install Playwright browsers
RUN playwright install chromium && \
    playwright install-deps chromium

# Clone WebArena repository (or copy if building with local version)
# We'll use environment variable to allow flexibility
ARG WEBARENA_REPO=https://github.com/web-arena-x/webarena.git
ARG WEBARENA_BRANCH=main
RUN git clone --depth 1 --branch ${WEBARENA_BRANCH} ${WEBARENA_REPO} /webarena

# Install WebArena
WORKDIR /webarena
RUN pip install --no-cache-dir -e .

# Copy OpenEnv core and webarena_env code
WORKDIR /app
COPY src/core/ /app/src/core/
COPY src/envs/webarena_env/ /app/src/envs/webarena_env/
COPY src/envs/webarena_env/README.md /app/README.md

# Set environment variables
ENV PYTHONPATH=/app/src:/webarena
ENV PYTHONUNBUFFERED=1
ENV WEBARENA_PATH=/webarena
ENV WEBARENA_HEADLESS=true
ENV WEBARENA_OBSERVATION_TYPE=accessibility_tree
ENV WEBARENA_VIEWPORT_WIDTH=1280
ENV WEBARENA_VIEWPORT_HEIGHT=720
ENV WEBARENA_CONFIG_DIR=/app/config_files

# Create config directory
RUN mkdir -p /app/config_files

# These should be set by the user when running the container:
# ENV SHOPPING=
# ENV SHOPPING_ADMIN=
# ENV REDDIT=
# ENV GITLAB=
# ENV MAP=
# ENV WIKIPEDIA=
# ENV HOMEPAGE=

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "envs.webarena_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
