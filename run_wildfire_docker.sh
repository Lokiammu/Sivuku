#!/bin/bash
# Script to run Wildfire Environment with Docker

set -e

WIDTH="${WILDFIRE_WIDTH:-32}"
HEIGHT="${WILDFIRE_HEIGHT:-32}"
HUMIDITY="${WILDFIRE_HUMIDITY:-0.25}"
PORT="${PORT:-8000}"
CONTAINER_NAME="wildfire-env-container"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Wildfire Environment - Docker Runner${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Step 1: Stop and remove any existing wildfire containers
echo -e "${YELLOW}Step 1: Cleaning up any existing containers...${NC}"
docker stop $(docker ps -aq --filter "name=wildfire") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=wildfire") 2>/dev/null || true
echo -e "${GREEN}✓ Cleaned up existing containers${NC}"
echo ""

# Step 2: Check if base image exists, if not build it
if ! docker images -q openenv-base:latest | grep -q .; then
    echo -e "${YELLOW}Step 2a: Building base image (openenv-base:latest)...${NC}"
    docker build -f src/core/containers/images/Dockerfile -t openenv-base:latest . > /dev/null
    echo -e "${GREEN}✓ Base image built successfully${NC}"
    echo ""
else
    echo -e "${GREEN}✓ Base image exists${NC}"
    echo ""
fi

# Step 3: Rebuild wildfire image to ensure latest code changes are included
echo -e "${YELLOW}Step 2b: Building Wildfire Docker image...${NC}"
docker build -f src/envs/wildfire_env/server/Dockerfile -t wildfire-env:latest . > /dev/null
echo -e "${GREEN}✓ Wildfire image built successfully${NC}"
echo ""

# Step 4: Start the container
echo -e "${BLUE}Step 4: Starting Wildfire Environment container...${NC}"
echo ""
echo "Configuration:"
echo "  Grid Width: $WIDTH"
echo "  Grid Height: $HEIGHT"
echo "  Humidity: $HUMIDITY"
echo "  Port: $PORT"
echo "  Web Interface: Enabled"
echo ""

docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:8000 \
  -e ENABLE_WEB_INTERFACE=true \
  -e WILDFIRE_WIDTH=$WIDTH \
  -e WILDFIRE_HEIGHT=$HEIGHT \
  -e WILDFIRE_HUMIDITY=$HUMIDITY \
  wildfire-env:latest > /dev/null

echo -e "${GREEN}✓ Container started successfully!${NC}"
echo ""

# Step 5: Wait a moment and check status
sleep 2
echo -e "${BLUE}Container Information:${NC}"
echo "  Name: $CONTAINER_NAME"
echo "  Status: $(docker ps -f name=$CONTAINER_NAME --format '{{.Status}}')"
echo ""

# Step 6: Display access information
echo -e "${GREEN}Web Interface: http://localhost:$PORT/web${NC}"
echo ""
echo "Available actions:"
echo "  - water: Apply water to a cell (extinguishes fire)"
echo "  - break: Create a firebreak (prevents fire spread)"
echo "  - wait: Do nothing (fire continues spreading)"
echo ""
echo -e "${BLUE}Showing logs (press Ctrl+C to stop):${NC}"
echo ""

# Step 7: Show logs
docker logs -f $CONTAINER_NAME
