FROM python:3.11-slim

# Install uv - a fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# ENV CELERY_BROKER_URL=redis://localhost:6379/0
# ENV CELERY_RESULT_BACKEND=redis://localhost:6379/0  

ENV CELERY_BROKER_URL=redis://:yourredispassword@redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://:yourredispassword@redis:6379/0


# Install system dependencies required by your packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the dependency file first to leverage Docker cache
COPY pyproject.toml .

# Install dependencies using uv
# This installs the project defined in pyproject.toml
# ADDED --system flag to install into the global site-packages
RUN uv pip install --system --no-cache .

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp

# Default command (will be overridden by your docker-compose file for each service)
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]
