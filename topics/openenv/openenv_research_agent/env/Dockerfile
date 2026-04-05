# OpenEnv Research Environment — Docker container
#
# This container runs the ResearchEnvironment as an OpenEnv HTTP server.
# Each episode (agent run) gets its own isolated container instance,
# which is OpenEnv's core isolation guarantee.
#
# Build:
#   docker build -t research-env:latest ./env
#
# Run locally (for development):
#   docker run -p 8000:8000 --env-file ../.env research-env:latest
#
# The OpenEnv client connects to http://localhost:8000

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer unless requirements change)
COPY requirements_env.txt .
RUN pip install --no-cache-dir -r requirements_env.txt

# Copy environment source
COPY env/ ./env/
COPY reward.py .

# OpenEnv server entrypoint — starts the FastAPI/WebSocket server
# that the EnvClient connects to via HTTP on port 8000
CMD ["python", "-m", "openenv.core.env_server.http_server", \
     "--env-class", "env.research_env:ResearchEnvironment", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
