#!/bin/bash

# OpenEnv Hugging Face Deployment Preparation Script
# This script prepares files for deployment to Hugging Face Spaces

set -e

ENV_NAME="$1"
BASE_IMAGE_SHA="$2"
STAGING_DIR="hf-staging"

if [ -z "$ENV_NAME" ]; then
    echo "Error: Environment name is required"
    exit 1
fi

# Handle "all" case by getting list of all available environments
if [ "$ENV_NAME" = "all" ]; then
    ENV_NAMES=$(ls -1 src/envs/ | grep -v README.md)
    echo "Detected 'all' - will process environments: $ENV_NAMES"
else
    ENV_NAMES="$ENV_NAME"
fi

# Set base image reference
if [ -n "$BASE_IMAGE_SHA" ]; then
    BASE_IMAGE_REF="openenv-base:$BASE_IMAGE_SHA"
    echo "Using specific SHA for openenv-base: $BASE_IMAGE_SHA"
else
    BASE_IMAGE_REF="openenv-base:latest"
    echo "Using latest tag for openenv-base"
fi

# Process each environment
for CURRENT_ENV in $ENV_NAMES; do
    echo "Preparing $CURRENT_ENV environment for deployment..."
    
    # Create staging directory for this environment
    CURRENT_STAGING_DIR="${STAGING_DIR}_${CURRENT_ENV}"
    mkdir -p $CURRENT_STAGING_DIR/src/core
    mkdir -p $CURRENT_STAGING_DIR/src/envs/$CURRENT_ENV

    # Copy core files
    cp -r src/core/* $CURRENT_STAGING_DIR/src/core/
    echo "Copied core files for $CURRENT_ENV"

    # Copy environment files
    cp -r src/envs/$CURRENT_ENV/* $CURRENT_STAGING_DIR/src/envs/$CURRENT_ENV/
    echo "Copied $CURRENT_ENV environment files"

# Create environment-specific multi-stage Dockerfile
create_environment_dockerfile() {
    local env_name=$1
    
    # Create base Dockerfile
    cat > $CURRENT_STAGING_DIR/Dockerfile << DOCKERFILE_EOF
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the specified openenv-base image
FROM $BASE_IMAGE_REF
DOCKERFILE_EOF

    # Add environment-specific dependencies
    case $env_name in
        "echo_env")
            # Echo environment needs no additional dependencies
            ;;
        "coding_env")
            cat >> $CURRENT_STAGING_DIR/Dockerfile << 'DOCKERFILE_EOF'
# Install smolagents for code execution
RUN pip install --no-cache-dir smolagents
DOCKERFILE_EOF
            ;;
        "chat_env")
            cat >> $CURRENT_STAGING_DIR/Dockerfile << 'DOCKERFILE_EOF'
# Install additional dependencies for ChatEnvironment
RUN pip install --no-cache-dir torch transformers

# Set up cache directory for Hugging Face models
RUN mkdir -p /.cache && chmod 777 /.cache
ENV HF_HOME=/.cache
ENV TRANSFORMERS_CACHE=/.cache

# Pre-download the GPT-2 model to avoid permission issues during runtime
RUN python -c "from transformers import GPT2Tokenizer; GPT2Tokenizer.from_pretrained('gpt2')"
DOCKERFILE_EOF
            ;;
        "atari_env")
            cat >> $CURRENT_STAGING_DIR/Dockerfile << 'DOCKERFILE_EOF'
# Install ALE-specific dependencies
RUN pip install --no-cache-dir \
    gymnasium>=0.29.0 \
    ale-py>=0.8.0 \
    numpy>=1.24.0
DOCKERFILE_EOF
            ;;
        "openspiel_env")
            # OpenSpiel requires special C++ build process - replace entire Dockerfile
            cat > $CURRENT_STAGING_DIR/Dockerfile << DOCKERFILE_EOF
# OpenSpiel requires complex C++ build - using special multi-stage approach
# Stage 1: Build OpenSpiel C++ bindings
FROM python:3.11 AS openspiel-builder

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set up OpenSpiel build directory
RUN mkdir /repo
WORKDIR /repo

# Clone OpenSpiel
RUN git clone https://github.com/google-deepmind/open_spiel.git .

# Run OpenSpiel's installation script (downloads C++ dependencies)
RUN ./install.sh

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade setuptools testresources importlib_metadata
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt cmake

# Build OpenSpiel with Python 3.11
RUN mkdir -p build
WORKDIR /repo/build
RUN cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=$(which clang++) ../open_spiel
RUN make -j$(nproc) pyspiel

# Stage 2: Use the specified openenv-base image
FROM $BASE_IMAGE_REF

# Copy OpenSpiel build artifacts from builder
RUN mkdir -p /repo
COPY --from=openspiel-builder /repo /repo

# Install OpenSpiel Python requirements in runtime
WORKDIR /repo
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# Set Python path for OpenSpiel
ENV PYTHONPATH=/repo:/repo/build/python:${PYTHONPATH}

# Copy OpenEnv core
WORKDIR /app
COPY src/core/ /app/src/core/
COPY src/envs/openspiel_env/ /app/src/envs/openspiel_env/

# Extend Python path for OpenEnv (base image set PYTHONPATH=/app/src)
# We prepend OpenSpiel paths
ENV PYTHONPATH=/repo:/repo/build/python:/app/src

# OpenSpiel-specific environment variables (can be overridden at runtime)
ENV OPENSPIEL_GAME=catch
ENV OPENSPIEL_AGENT_PLAYER=0
ENV OPENSPIEL_OPPONENT_POLICY=random

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.openspiel_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE_EOF
            echo "Created special OpenSpiel Dockerfile with C++ build process"
            echo "OpenSpiel builds can take 10-15 minutes due to C++ compilation"
            return  # Skip the common parts since OpenSpiel has its own complete Dockerfile
            ;;
    esac

    # Add common parts
    cat >> $CURRENT_STAGING_DIR/Dockerfile << 'DOCKERFILE_EOF'

# Copy only what's needed for this environment
COPY src/core/ /app/src/core/
COPY src/envs/ENV_NAME_PLACEHOLDER/ /app/src/envs/ENV_NAME_PLACEHOLDER/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.ENV_NAME_PLACEHOLDER.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE_EOF

    # Replace placeholder with actual environment name
    sed -i "s/ENV_NAME_PLACEHOLDER/$env_name/g" $CURRENT_STAGING_DIR/Dockerfile
}

    create_environment_dockerfile $CURRENT_ENV

    # Add web interface support
    echo "ENV ENABLE_WEB_INTERFACE=true" >> $CURRENT_STAGING_DIR/Dockerfile
    echo "Added web interface support for $CURRENT_ENV"

    # Create environment-specific README
create_readme() {
    local env_name=$1
    
    # Capitalize first letter of environment name
    env_title=$(echo "$env_name" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')
    
    cat > $CURRENT_STAGING_DIR/README.md << README_EOF
---
title: ${env_title} Environment Server
emoji: 🐳
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
---

# ${env_title} Environment Server

FastAPI server for ${env_name} environment powered by Meta's OpenEnv.

## About

This Space provides a containerized environment for ${env_name} interactions.
Built with FastAPI and OpenEnv framework.

## Web Interface

This deployment includes an interactive web interface for exploring the environment:
- **HumanAgent Interface**: Interact with the environment using a web form
- **State Observer**: Real-time view of environment state and action history
- **Live Updates**: WebSocket-based real-time updates

Access the web interface at: \`/web\`

README_EOF

    # Add environment-specific information
    case $env_name in
        "echo_env")
            cat >> $CURRENT_STAGING_DIR/README.md << 'README_EOF'
## Echo Environment

Simple test environment that echoes back messages. Perfect for testing the OpenEnv APIs.

### Usage
Send a POST request to `/step` with:
```json
{
  "message": "Hello World"
}
```
README_EOF
            ;;
        "coding_env")
            cat >> $CURRENT_STAGING_DIR/README.md << 'README_EOF'
## Coding Environment

Executes Python code in a sandboxed environment with safety checks.

### Usage
Send a POST request to `/step` with:
```json
{
  "code": "print('Hello World')"
}
```
README_EOF
            ;;
        "chat_env")
            cat >> $CURRENT_STAGING_DIR/README.md << 'README_EOF'
## Chat Environment

Provides a chat-based interface for LLMs with tokenization support.

### Usage
Send a POST request to `/step` with tokenized input:
```json
{
  "tokens": [1, 2, 3, 4, 5]
}
```
README_EOF
            ;;
        "atari_env")
            cat >> $CURRENT_STAGING_DIR/README.md << 'README_EOF'
## Atari Environment

Provides Atari 2600 games via the Arcade Learning Environment (ALE).

### Usage
Send a POST request to `/step` with:
```json
{
  "action_id": 0,
  "game_name": "pong"
}
```
README_EOF
            ;;
        "openspiel_env")
            cat >> $CURRENT_STAGING_DIR/README.md << 'README_EOF'
## OpenSpiel Environment

Provides access to OpenSpiel games for multi-agent reinforcement learning.

### Usage
Send a POST request to `/step` with:
```json
{
  "action": 0
}
```
README_EOF
            ;;
    esac

    cat >> $CURRENT_STAGING_DIR/README.md << 'README_EOF'

## API Documentation

Visit `/docs` for interactive API documentation.

## Health Check

The environment provides a health check endpoint at `/health`.
README_EOF
}

    create_readme $CURRENT_ENV
    echo "Created README for HF Space for $CURRENT_ENV"
    echo "Completed preparation for $CURRENT_ENV environment"
done

echo "All environments prepared successfully!"
