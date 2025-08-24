# server_celery.py
"""
Updated FastAPI server with Celery task queue integration
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import tempfile
import shutil
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import redis
from celery.result import AsyncResult
from tasks import (
    process_video_task,
    process_video_high_priority_task,
    process_batch_task,
    get_task_progress,
    app as celery_app
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Async Video Processing API with Celery",
    description="Production-ready video transcription and optimization service with task queue",
    version="3.0.0"
)

# Initialize Redis client for task metadata
redis_client = redis.StrictRedis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0)),
    password=os.getenv('REDIS_PASSWORD', None),
    decode_responses=True
)

class ProcessingRequest(BaseModel):
    input_video_path: str
    user_id: str
    output_video_path: Optional[str] = None
    keep_intermediate_files: bool = False
    config: Dict[str, Any] = {}
    priority: str = "normal"  # "normal" or "high"

class BatchProcessingRequest(BaseModel):
    requests: List[ProcessingRequest]
    max_concurrent: int = 2

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None
    current_step: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    queue: str


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Video Processing API with Celery")
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        logger.warning("Task tracking may be limited without Redis")
    
    # Check Celery workers
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        if stats:
            logger.info(f"Found {len(stats)} active Celery workers")
        else:
            logger.warning("No active Celery workers found. Tasks will be queued.")
    except Exception as e:
        logger.error(f"Could not inspect Celery workers: {e}")


@app.get("/")
async def root():
    """Health check and system information endpoint"""
    try:
        # Get system health from Redis
        health_data = redis_client.hgetall("system:health") or {}
        
        # Get queue sizes
        inspect = celery_app.control.inspect()
        active_tasks = reserved_tasks = 0
        
        if inspect:
            active = inspect.active()
            if active:
                active_tasks = sum(len(tasks) for tasks in active.values())
            
            reserved = inspect.reserved()
            if reserved:
                reserved_tasks = sum(len(tasks) for tasks in reserved.values())
        
        return {
            "message": "Video Processing API with Celery",
            "status": "running",
            "version": "3.0.0",
            "features": [
                "Distributed task processing with Celery",
                "Redis-based task queue and result storage",
                "Real-time progress tracking",
                "Priority queues for premium users",
                "Automatic retry with exponential backoff",
                "Task monitoring and health checks"
            ],
            "system_status": {
                "redis_connected": redis_client.ping(),
                "active_tasks": active_tasks,
                "queued_tasks": reserved_tasks,
                "last_health_check": health_data.get("timestamp", "N/A")
            },
            "endpoints": {
                "process": "/process-video/",
                "upload": "/upload-and-process/",
                "batch": "/process-batch/",
                "status": "/task/{task_id}",
                "cancel": "/task/{task_id}/cancel",
                "all_tasks": "/tasks/",
                "user_tasks": "/users/{user_id}/tasks/",
                "queue_stats": "/queues/stats",
                "health": "/health"
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {"message": "API running", "status": "degraded", "error": str(e)}


@app.post("/process-video/", response_model=TaskResponse)
async def process_video_endpoint(request: ProcessingRequest):
    """
    Queue a video processing task
    """
    if not os.path.exists(request.input_video_path):
        raise HTTPException(status_code=404, detail="Input video file not found")
    
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Choose the appropriate task based on priority
    if request.priority == "high":
        task = process_video_high_priority_task.apply_async(
            args=[
                request.input_video_path,
                request.user_id,
                request.output_video_path,
                request.keep_intermediate_files,
                request.config
            ],
            queue='high_priority'
        )
        queue_name = "high_priority"
    else:
        task = process_video_task.apply_async(
            args=[
                request.input_video_path,
                request.user_id,
                request.output_video_path,
                request.keep_intermediate_files,
                request.config
            ],
            queue='video_processing'
        )
        queue_name = "video_processing"
    
    # Store additional metadata in Redis
    redis_client.hset(f"task:{task.id}", mapping={
        "user_id": request.user_id,
        "input_video": request.input_video_path,
        "priority": request.priority,
        "queue": queue_name,
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending"
    })
    redis_client.expire(f"task:{task.id}", 86400)  # Expire after 24 hours
    
    logger.info(f"Queued task {task.id} for user {request.user_id} in {queue_name} queue")
    
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message=f"Video processing task queued successfully",
        queue=queue_name
    )


@app.post("/upload-and-process/", response_model=TaskResponse)
async def upload_and_process_video(
    user_id: str = Form(...),
    video_file: UploadFile = File(...),
    keep_intermediate_files: bool = Form(False),
    config: str = Form("{}"),
    priority: str = Form("normal")
):
    """
    Upload a video file and queue it for processing
    """
    import json
    
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    
    # Parse config JSON
    try:
        config_dict = json.loads(config) if config else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in config parameter")
    
    # Validate file type
    if not video_file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save uploaded file to a persistent location
    upload_dir = os.getenv('UPLOAD_DIR', '/tmp/uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    import uuid
    file_extension = os.path.splitext(video_file.filename)[1]
    unique_filename = f"{user_id}_{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save file
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(video_file.file, f)
    
    logger.info(f"Uploaded video saved to: {file_path}")
    
    # Queue the task
    if priority == "high":
        task = process_video_high_priority_task.apply_async(
            args=[file_path, user_id, None, keep_intermediate_files, config_dict],
            queue='high_priority'
        )
        queue_name = "high_priority"
    else:
        task = process_video_task.apply_async(
            args=[file_path, user_id, None, keep_intermediate_files, config_dict],
            queue='video_processing'
        )
        queue_name = "video_processing"
    
    # Store metadata
    redis_client.hset(f"task:{task.id}", mapping={
        "user_id": user_id,
        "input_video": file_path,
        "original_filename": video_file.filename,
        "priority": priority,
        "queue": queue_name,
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending",
        "is_uploaded": "true"
    })
    redis_client.expire(f"task:{task.id}", 86400)
    
    return TaskResponse(
        task_id=task.id,
        status="queued",
        message=f"Video uploaded and queued for processing",
        queue=queue_name
    )


@app.post("/process-batch/", response_model=Dict[str, Any])
async def process_batch_endpoint(request: BatchProcessingRequest):
    """
    Queue multiple videos for batch processing
    """
    if len(request.requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 videos per batch")
    
    # Validate all requests
    for req in request.requests:
        if not os.path.exists(req.input_video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Input video file not found: {req.input_video_path}"
            )
        if not req.user_id or not req.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required for all requests")
    
    # Prepare batch data
    video_requests = [
        {
            'input_path': req.input_video_path,
            'user_id': req.user_id,
            'output_path': req.output_video_path,
            'keep_intermediate_files': req.keep_intermediate_files,
            'config': req.config
        }
        for req in request.requests
    ]
    
    # Queue batch task
    batch_task = process_batch_task.apply_async(
        args=[video_requests, request.max_concurrent],
        queue='batch_processing'
    )
    
    return {
        "batch_id": batch_task.id,
        "status": "queued",
        "total_videos": len(request.requests),
        "message": f"Batch of {len(request.requests)} videos queued for processing"
    }


@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """
    Get the status and progress of a specific task
    """
    # Get task info from Redis
    task_data = redis_client.hgetall(f"task:{task_id}")
    
    # Get Celery task result
    result = AsyncResult(task_id, app=celery_app)
    
    if not task_data and result.state == 'PENDING':
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Merge data from Redis and Celery
    status = result.state
    
    response_data = {
        "task_id": task_id,
        "status": status,
        "progress": None,
        "current_step": None,
        "result": None,
        "error": None,
        "created_at": task_data.get("created_at"),
        "completed_at": task_data.get("completed_at")
    }
    
    if status == 'PENDING':
        response_data["status"] = "queued"
    elif status == 'PROGRESS':
        info = result.info or {}
        response_data["progress"] = info.get('current', 0)
        response_data["current_step"] = info.get('step', '')
    elif status == 'SUCCESS':
        response_data["status"] = "completed"
        response_data["result"] = result.result
    elif status == 'FAILURE':
        response_data["status"] = "failed"
        response_data["error"] = str(result.info)
    elif status == 'RETRY':
        response_data["status"] = "retrying"
    
    # Override with Redis data if available
    if task_data:
        response_data["progress"] = int(task_data.get("progress", 0))
        response_data["current_step"] = task_data.get("current_step", "")
        if task_data.get("gdrive_file_id"):
            response_data["result"] = response_data.get("result", {})
            response_data["result"]["gdrive_file_id"] = task_data["gdrive_file_id"]
            response_data["result"]["gdrive_link"] = task_data.get("gdrive_link", "")
    
    return TaskStatus(**response_data)


@app.delete("/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    Cancel a pending or running task
    """
    result = AsyncResult(task_id, app=celery_app)
    
    if result.state == 'PENDING':
        # Task hasn't started yet
        result.revoke(terminate=False)
        message = "Task cancelled (was queued)"
    elif result.state in ['PROGRESS', 'RETRY']:
        # Task is running - terminate it
        result.revoke(terminate=True, signal='SIGKILL')
        message = "Task terminated (was running)"
    elif result.state in ['SUCCESS', 'FAILURE']:
        raise HTTPException(status_code=400, detail="Cannot cancel completed task")
    else:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update Redis
    redis_client.hset(f"task:{task_id}", mapping={
        "status": "cancelled",
        "cancelled_at": datetime.utcnow().isoformat()
    })
    
    return {"task_id": task_id, "status": "cancelled", "message": message}


@app.get("/tasks/")
async def list_all_tasks(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None
):
    """
    List all tasks with optional filtering
    """
    tasks = []
    cursor = 0
    count = 0
    
    # Scan Redis for task keys
    while True:
        cursor, keys = redis_client.scan(
            cursor,
            match="task:*",
            count=100
        )
        
        for key in keys:
            if count >= offset and len(tasks) < limit:
                task_data = redis_client.hgetall(key)
                task_id = key.split(":", 1)[1]
                
                # Filter by status if specified
                if status and task_data.get("status") != status:
                    continue
                
                # Get Celery status
                result = AsyncResult(task_id, app=celery_app)
                
                tasks.append({
                    "task_id": task_id,
                    "user_id": task_data.get("user_id"),
                    "status": result.state if result.state != 'PENDING' else task_data.get("status", "unknown"),
                    "priority": task_data.get("priority", "normal"),
                    "created_at": task_data.get("created_at"),
                    "progress": task_data.get("progress", "0")
                })
            
            count += 1
            
            if len(tasks) >= limit:
                break
        
        if cursor == 0 or len(tasks) >= limit:
            break
    
    return {
        "total_returned": len(tasks),
        "limit": limit,
        "offset": offset,
        "tasks": tasks
    }


@app.get("/users/{user_id}/tasks/")
async def get_user_tasks(
    user_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None
):
    """
    Get all tasks for a specific user
    """
    user_tasks = []
    cursor = 0
    
    # Scan Redis for user's tasks
    while True:
        cursor, keys = redis_client.scan(
            cursor,
            match="task:*",
            count=100
        )
        
        for key in keys:
            task_data = redis_client.hgetall(key)
            
            if task_data.get("user_id") == user_id:
                task_id = key.split(":", 1)[1]
                
                # Filter by status if specified
                if status and task_data.get("status") != status:
                    continue
                
                # Get Celery status
                result = AsyncResult(task_id, app=celery_app)
                
                user_tasks.append({
                    "task_id": task_id,
                    "status": result.state if result.state != 'PENDING' else task_data.get("status", "unknown"),
                    "input_video": task_data.get("input_video"),
                    "priority": task_data.get("priority", "normal"),
                    "created_at": task_data.get("created_at"),
                    "completed_at": task_data.get("completed_at"),
                    "progress": task_data.get("progress", "0"),
                    "gdrive_link": task_data.get("gdrive_link")
                })
        
        if cursor == 0:
            break
    
    # Sort by created_at descending
    user_tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    # Apply pagination
    paginated_tasks = user_tasks[offset:offset + limit]
    
    return {
        "user_id": user_id,
        "total_tasks": len(user_tasks),
        "total_returned": len(paginated_tasks),
        "limit": limit,
        "offset": offset,
        "tasks": paginated_tasks
    }


@app.get("/queues/stats")
async def get_queue_statistics():
    """
    Get statistics about all queues
    """
    try:
        inspect = celery_app.control.inspect()
        
        stats = {
            "queues": {
                "video_processing": {"active": 0, "reserved": 0},
                "high_priority": {"active": 0, "reserved": 0},
                "batch_processing": {"active": 0, "reserved": 0}
            },
            "workers": [],
            "total_active": 0,
            "total_reserved": 0
        }
        
        # Get active tasks
        active = inspect.active()
        if active:
            for worker, tasks in active.items():
                stats["workers"].append({
                    "name": worker,
                    "active_tasks": len(tasks)
                })
                stats["total_active"] += len(tasks)
                
                # Categorize by queue
                for task in tasks:
                    queue = task.get("delivery_info", {}).get("routing_key", "video_processing")
                    if queue in stats["queues"]:
                        stats["queues"][queue]["active"] += 1
        
        # Get reserved tasks
        reserved = inspect.reserved()
        if reserved:
            for worker, tasks in reserved.items():
                stats["total_reserved"] += len(tasks)
                
                # Categorize by queue
                for task in tasks:
                    queue = task.get("delivery_info", {}).get("routing_key", "video_processing")
                    if queue in stats["queues"]:
                        stats["queues"][queue]["reserved"] += 1
        
        # Get Redis queue lengths (approximate)
        for queue_name in stats["queues"]:
            queue_key = f"celery:{queue_name}"
            queue_length = redis_client.llen(queue_key)
            stats["queues"][queue_name]["waiting"] = queue_length
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting queue statistics: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve queue statistics")


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "redis": False,
            "celery_workers": False,
            "mongodb": False
        },
        "issues": []
    }
    
    # Check Redis
    try:
        redis_client.ping()
        health["checks"]["redis"] = True
    except Exception as e:
        health["issues"].append(f"Redis unavailable: {e}")
        health["status"] = "degraded"
    
    # Check Celery workers
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        if stats and len(stats) > 0:
            health["checks"]["celery_workers"] = True
            health["worker_count"] = len(stats)
        else:
            health["issues"].append("No Celery workers available")
            health["status"] = "degraded"
    except Exception as e:
        health["issues"].append(f"Cannot inspect Celery workers: {e}")
        health["status"] = "degraded"
    
    # Check MongoDB (optional)
    try:
        from mongodb_manager import MongoDBManager
        mongo_uri = os.getenv('MONGODB_URI')
        if mongo_uri:
            # This is a simplified check - implement proper health check in MongoDBManager
            health["checks"]["mongodb"] = True
    except Exception:
        health["issues"].append("MongoDB connection not configured")
    
    # Get system health from Redis
    system_health = redis_client.hgetall("system:health")
    if system_health:
        health["last_monitor_run"] = system_health.get("timestamp")
        health["metrics"] = {
            "active_tasks": system_health.get("active_tasks", 0),
            "pending_tasks": system_health.get("pending_tasks", 0),
            "failed_tasks": system_health.get("failed_tasks", 0),
            "completed_tasks": system_health.get("completed_tasks", 0)
        }
    
    if health["status"] == "degraded" and all(not check for check in health["checks"].values()):
        health["status"] = "unhealthy"
    
    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_celery:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )