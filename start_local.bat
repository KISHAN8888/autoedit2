@echo off
echo Starting Video Processing API locally...

echo Starting Redis...
start "Redis" redis-server

echo Starting MongoDB...
start "MongoDB" mongod

timeout /t 5

echo Starting API...
start "API" python -m uvicorn app.main:app --host localhost --port 8000 --reload

echo Starting Processing Worker...
start "Processing Worker" celery -A app.workers.celery_app worker --loglevel=info --queues=processing --concurrency=2

echo Starting FFmpeg Worker...
start "FFmpeg Worker" celery -A app.workers.celery_app worker --loglevel=info --queues=ffmpeg --concurrency=1

echo All services started! Check http://localhost:8000/docs
pause