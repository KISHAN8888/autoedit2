#!/bin/bash

# Function to start API
start_api() {
    echo "Starting FastAPI application..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
}

# Function to start processing worker
start_processing_worker() {
    echo "Starting Celery processing worker..."
    exec celery -A app.workers.celery_app worker --loglevel=info --queues=processing --concurrency=2
}

# Function to start FFmpeg worker
start_ffmpeg_worker() {
    echo "Starting Celery FFmpeg worker..."
    exec celery -A app.workers.celery_app worker --loglevel=info --queues=ffmpeg --concurrency=1
}

# Check command and start appropriate service
case "$1" in
    "api")
        start_api
        ;;
    "processing-worker")
        start_processing_worker
        ;;
    "ffmpeg-worker")
        start_ffmpeg_worker
        ;;
    *)
        echo "Usage: $0 {api|processing-worker|ffmpeg-worker}"
        exit 1
        ;;
esac
