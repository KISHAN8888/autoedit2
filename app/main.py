from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import os
import uuid
import tempfile
import shutil
from pathlib import Path
import logging
from datetime import datetime
import json

from app.models import TaskResponse, TaskStatus
from app.workers.celery_app import celery_app
from app.database.mongodb_manager import MongoDBManager
from app.config import get_settings
from app.middleware.security import RateLimitMiddleware, SecurityHeadersMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Processing API",
    description="Two-worker async video processing service",
    version="2.0.0"
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, calls=50, period=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

settings = get_settings()
mongodb_manager = None
MAX_FILE_SIZE = 500 * 1024 * 1024

@app.on_event("startup")
async def startup_event():
    global mongodb_manager
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    mongodb_manager = MongoDBManager(
        mongodb_uri=settings.mongodb_uri,
        database_name=settings.mongodb_database
    )
    await mongodb_manager.initialize()
    logger.info("Video Processing API with dual workers started")

@app.on_event("shutdown")
async def shutdown_event():
    global mongodb_manager
    if mongodb_manager:
        await mongodb_manager.close()
    logger.info("Application shutdown complete")

@app.get("/health")
async def health_check():
    try:
        await mongodb_manager.client.admin.command('ping')
        mongo_status = "healthy"
    except:
        mongo_status = "unhealthy"
    
    inspect = celery_app.control.inspect()
    workers = inspect.active()
    worker_count = len(workers) if workers else 0
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "mongodb": mongo_status,
            "celery_workers": worker_count
        }
    }

@app.post("/process-video", response_model=TaskResponse)
async def process_video(
    user_id: str,
    file: UploadFile = File(...),
    background_image_path: Optional[str] = None,
    overlay_options: Optional[str] = None
):
    """Submit video for processing - goes directly to processing worker"""
    
    # Validate file
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    task_id = str(uuid.uuid4())
    upload_dir = f"uploads/{task_id}"
    os.makedirs(upload_dir, exist_ok=True)
    input_path = os.path.join(upload_dir, f"input_{file.filename}")
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(input_path)
        logger.info(f"File uploaded: {file.filename} ({file_size} bytes)")
        
        # Parse config
        config = {}
        if background_image_path:
            config["background_image_path"] = background_image_path
        
        if overlay_options:
            try:
                config["overlay_options"] = json.loads(overlay_options)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid overlay_options JSON")
        
        # Create initial task record
        initial_task_data = {
            "task_id": task_id,
            "user_id": user_id,
            "filename": file.filename,
            "file_size": file_size,
            "status": "received",
            "created_at": datetime.utcnow(),
            "input_path": input_path,
            "upload_dir": upload_dir,
            "config": config
        }
        
        await mongodb_manager.create_task_record(initial_task_data)
        
        # Submit DIRECTLY to processing worker (high concurrency, no queuing)
        processing_task = celery_app.send_task(
            'app.workers.processing_worker.process_video_content',
            args=[task_id, input_path, user_id, config],
            task_id=f"processing_{task_id}",
            queue='processing'  # Different queue for processing worker
        )
        
        await mongodb_manager.update_task_data(task_id, {
            "processing_task_id": processing_task.id,
            "status": "processing"
        })
        
        logger.info(f"Task {task_id} submitted to processing worker: {processing_task.id}")
        
        return TaskResponse(
            task_id=task_id,
            status="processing",
            message="Video processing started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir, ignore_errors=True)
        
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/task/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get comprehensive task status"""
    task_data = await mongodb_manager.get_task_by_id(task_id)
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Add Celery task info for both workers if available
    if task_data.get("processing_task_id"):
        processing_result = celery_app.AsyncResult(task_data["processing_task_id"])
        task_data["processing_status"] = processing_result.status
        if hasattr(processing_result, 'info') and processing_result.info:
            task_data["processing_progress"] = processing_result.info.get("progress")
    
    if task_data.get("ffmpeg_task_id"):
        ffmpeg_result = celery_app.AsyncResult(task_data["ffmpeg_task_id"])
        task_data["ffmpeg_status"] = ffmpeg_result.status
        if hasattr(ffmpeg_result, 'info') and ffmpeg_result.info:
            task_data["ffmpeg_progress"] = ffmpeg_result.info.get("progress")
    
    return TaskStatus(**task_data)

@app.get("/user/{user_id}/tasks")
async def get_user_tasks(user_id: str, limit: int = 20, offset: int = 0):
    """Get user's tasks with pagination"""
    if limit > 100:
        limit = 100
        
    tasks = await mongodb_manager.get_user_tasks(user_id, limit, offset)
    total_count = await mongodb_manager.count_user_tasks(user_id)
    
    return {
        "tasks": tasks,
        "total": total_count,
        "limit": limit,
        "offset": offset
    }

@app.delete("/task/{task_id}")
async def cancel_task(task_id: str, user_id: str):
    """Cancel task - both workers if needed"""
    task_data = await mongodb_manager.get_task_by_id(task_id)
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_data["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Cancel both worker tasks if they exist
    if task_data.get("processing_task_id"):
        celery_app.control.revoke(task_data["processing_task_id"], terminate=True)
    
    if task_data.get("ffmpeg_task_id"):
        celery_app.control.revoke(task_data["ffmpeg_task_id"], terminate=True)
    
    await mongodb_manager.update_task_status(task_id, "cancelled")
    
    # Cleanup files
    if task_data.get("upload_dir") and os.path.exists(task_data["upload_dir"]):
        shutil.rmtree(task_data["upload_dir"], ignore_errors=True)
    
    return {"message": "Task cancelled successfully"}

@app.get("/stats")
async def get_system_stats():
    """System statistics endpoint"""
    return await mongodb_manager.get_processing_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=False)    

