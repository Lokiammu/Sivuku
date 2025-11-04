#!/bin/bash
# Script to test wildfire environment locally without Docker

echo "🔥 Wildfire Environment - Local Testing"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

# Set environment variables
export ENABLE_WEB_INTERFACE=true
export PYTHONPATH=src
export WILDFIRE_WIDTH=${WILDFIRE_WIDTH:-16}
export WILDFIRE_HEIGHT=${WILDFIRE_HEIGHT:-16}

echo "Configuration:"
echo "  Grid Width: $WILDFIRE_WIDTH"
echo "  Grid Height: $WILDFIRE_HEIGHT"
echo "  Web Interface: Enabled"
echo ""

# Check if we're in a virtual environment or if .venv exists
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python3"
    PIP_CMD="$VIRTUAL_ENV/bin/pip"
    echo "✅ Using virtual environment: $VIRTUAL_ENV"
elif [ -f "$(dirname "$0")/../../../../.venv/bin/python3" ]; then
    # Check for .venv in project root
    VENV_PATH="$(cd "$(dirname "$0")/../../.." && pwd)/.venv"
    PYTHON_CMD="$VENV_PATH/bin/python3"
    PIP_CMD="$VENV_PATH/bin/pip"
    echo "✅ Using project virtual environment: $VENV_PATH"
    export VIRTUAL_ENV="$VENV_PATH"
    export PATH="$VENV_PATH/bin:$PATH"
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# Check if uvicorn is installed
if ! $PYTHON_CMD -c "import uvicorn" 2>/dev/null; then
    echo "⚠️  uvicorn not found. Installing..."
    $PIP_CMD install uvicorn fastapi
fi

# Check if fastapi is installed
if ! $PYTHON_CMD -c "import fastapi" 2>/dev/null; then
    echo "⚠️  fastapi not found. Installing..."
    $PIP_CMD install fastapi
fi

echo ""
# Check if port 8000 is in use
PORT=8000
if lsof -ti:$PORT > /dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use!"
    echo ""
    # Check what's using it
    PROCESS=$(lsof -ti:$PORT | head -1)
    PROCESS_INFO=$(ps -p $PROCESS -o comm= 2>/dev/null || echo "unknown")
    echo "   Port $PORT is being used by: $PROCESS_INFO (PID: $PROCESS)"
    echo ""
    echo "Options:"
    echo "  1. Use a different port (8001)"
    echo "  2. Kill existing processes on port $PORT (⚠️  WARNING: May kill important processes)"
    echo ""
    read -p "Choose option (1 or 2, default 1): " choice
    choice=${choice:-1}
    
    if [ "$choice" = "1" ]; then
        PORT=8001
        echo "✅ Using port $PORT instead"
    else
        echo "⚠️  Killing processes on port $PORT..."
        lsof -ti:$PORT | xargs kill -9 2>/dev/null
        sleep 1
        if lsof -ti:$PORT > /dev/null 2>&1; then
            echo "❌ Failed to free port $PORT, using port 8001 instead"
            PORT=8001
        else
            echo "✅ Port $PORT is now free"
        fi
    fi
fi

echo ""
echo "🚀 Starting server..."
echo "   Access at: http://localhost:$PORT/web"
echo "   Press Ctrl+C to stop"
echo ""

# Run the server
cd "$(dirname "$0")/../../.."

# Ensure PYTHONPATH is set for uvicorn (needed for reload mode)
# The issue is that uvicorn's reload mode spawns a new process that needs PYTHONPATH
# We need to set it in a way that's inherited by the subprocess
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Use python -m uvicorn to ensure PYTHONPATH is respected in reload mode
$PYTHON_CMD -m uvicorn envs.wildfire_env.server.app:app --reload --host 0.0.0.0 --port $PORT

