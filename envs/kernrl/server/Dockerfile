# GPU-enabled Dockerfile for kernrl environment
# Requires NVIDIA GPU runtime

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV KERNRL_PROBLEMS_DIR=/app/env/problems

WORKDIR /app/env

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Copy environment files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install PyTorch with CUDA support and Triton
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir triton

# Expose port
EXPOSE 8000

# Set environment variables
ENV ENABLE_WEB_INTERFACE=true

# Run the server
CMD ["python", "-m", "uvicorn", "kernrl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
