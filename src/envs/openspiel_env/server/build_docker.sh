#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to build the OpenSpiel environment Docker image
# Usage: ./build_docker.sh [tag]
#
# Note: Requires envtorch-base:latest to be built first.
# See: src/core/containers/images/README.md

set -e

TAG="${1:-latest}"
IMAGE_NAME="openspiel-env:${TAG}"

echo "🐳 Building OpenSpiel Environment Docker Image"
echo "================================================"
echo "Image: $IMAGE_NAME"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to OpenEnv root (4 levels up from server/)
OPENENV_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "📁 OpenEnv root: $OPENENV_ROOT"
echo ""

# Check if envtorch-base image exists
if ! docker image inspect envtorch-base:latest >/dev/null 2>&1; then
    echo "❌ Base image 'envtorch-base:latest' not found!"
    echo ""
    echo "This environment requires the envtorch-base image."
    echo "Please build it first:"
    echo ""
    echo "  cd $OPENENV_ROOT"
    echo "  docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile ."
    echo ""
    echo "See src/core/containers/images/README.md for more details."
    echo ""
    exit 1
fi

echo "✓ Base image 'envtorch-base:latest' found"
echo ""

# Build OpenSpiel environment image
echo "⏳ Building (this may take 5-10 minutes due to OpenSpiel compilation)..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$OPENENV_ROOT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "🚀 Run with different games:"
    echo ""
    echo "  # Catch (default)"
    echo "  docker run -p 8000:8000 $IMAGE_NAME"
    echo ""
    echo "  # Tic-Tac-Toe"
    echo "  docker run -p 8000:8000 -e OPENSPIEL_GAME=tic_tac_toe $IMAGE_NAME"
    echo ""
    echo "  # Kuhn Poker"
    echo "  docker run -p 8000:8000 -e OPENSPIEL_GAME=kuhn_poker $IMAGE_NAME"
    echo ""
    echo "  # Cliff Walking"
    echo "  docker run -p 8000:8000 -e OPENSPIEL_GAME=cliff_walking $IMAGE_NAME"
    echo ""
    echo "  # 2048"
    echo "  docker run -p 8000:8000 -e OPENSPIEL_GAME=2048 $IMAGE_NAME"
    echo ""
    echo "  # Blackjack"
    echo "  docker run -p 8000:8000 -e OPENSPIEL_GAME=blackjack $IMAGE_NAME"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi
