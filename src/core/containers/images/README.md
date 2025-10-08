# EnvTorch Base Container Images

This directory contains base container image Dockerfiles used by all EnvTorch environment servers.

## Why Base Images?

Using standardized base images prevents **image cardinality explosion** when multiple environments are created. Instead of each environment building its own complete image from scratch, they all share a common base.

### Benefits

✅ **Reduced build times** - Environments only copy their specific files
✅ **Smaller total storage** - Base layers are shared across all environments
✅ **Consistent dependencies** - All environments use the same FastAPI/uvicorn versions
✅ **Easier maintenance** - Update dependencies in one place
✅ **Faster deployments** - Base image can be pre-pulled on servers

### Without Base Images (❌ Problem)
```
echo-env:latest        500 MB  (python + fastapi + uvicorn + app)
coding-env:latest      520 MB  (python + fastapi + uvicorn + app + tools)
another-env:latest     510 MB  (python + fastapi + uvicorn + app)
---
Total: 1.5 GB (with lots of duplication)
```

### With Base Images (✅ Solution)
```
envtorch-base:latest   300 MB  (python + fastapi + uvicorn)
echo-env:latest         50 MB  (app only, uses base)
coding-env:latest       70 MB  (app + tools, uses base)
another-env:latest      45 MB  (app only, uses base)
---
Total: 465 MB (base shared, minimal duplication)
```

## Building the Base Image

```bash
# From project root
docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .
```

## Usage in Environment Dockerfiles

Each environment Dockerfile should start with:

```dockerfile
FROM envtorch-base:latest

# Copy only environment-specific files
COPY src/core/ /app/src/core/
COPY src/envs/my_env/ /app/src/envs/my_env/

# Run the server
CMD ["uvicorn", "envs.my_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Base Image Contents

- Python 3.11-slim
- FastAPI >= 0.104.0
- Uvicorn >= 0.24.0
- Requests >= 2.25.0
- curl (for health checks)

## Example: Building Echo Environment

```bash
# Step 1: Build base image (do this once)
docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .

# Step 2: Build echo environment (uses base)
docker build -t echo-env:latest -f src/envs/echo_env/server/Dockerfile .

# Step 3: Run echo environment
docker run -p 8000:8000 echo-env:latest
```

## Updating the Base

When dependencies need updating:

1. Update `src/core/containers/images/Dockerfile`
2. Rebuild base image
3. Rebuild all environment images (they'll use new base)

```bash
# Update base
docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .

# Rebuild environments (they automatically use new base)
docker build -t echo-env:latest -f src/envs/echo_env/server/Dockerfile .
```
