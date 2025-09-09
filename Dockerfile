# # Use Python 3.11 slim image
# FROM python:3.11-slim

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/app
# ENV PYTHONDONTWRITEBYTECODE=1

# # Install system dependencies including FFmpeg
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Set work directory
# WORKDIR /app

# # Copy requirements first for better Docker layer caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy the entire project (including app/ directory)
# COPY . .

# # Create necessary directories
# RUN mkdir -p uploads logs outputs

# # Make entrypoint script executable
# RUN chmod +x entrypoint.sh

# # Expose port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Use entrypoint script
# ENTRYPOINT ["./entrypoint.sh"]


# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
# Add uv to the system's PATH
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies (like FFmpeg) and the uv installer in a single layer
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Set work directory
WORKDIR /app

# Copy the pyproject.toml file first to leverage Docker layer caching
# The build will only re-install dependencies if this file changes.
COPY pyproject.toml .

# Install Python dependencies from pyproject.toml using uv
# --system installs packages into the global site-packages, which is standard for Docker.
RUN uv pip install --no-cache --system .

# Copy the rest of the project files
COPY . .

# Create necessary directories for the application
RUN mkdir -p uploads logs outputs

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Expose the application port
EXPOSE 8000

# Add a health check to ensure the application is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the entrypoint for the container
ENTRYPOINT ["./entrypoint.sh"]

