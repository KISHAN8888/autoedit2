# # tasks.py (Updated Redis connection section)
# """
# Celery tasks with Redis Cloud connection
# """
# import os
# import asyncio
# import logging
# import tempfile
# import shutil
# import time
# from typing import Dict, List, Any, Optional
# from datetime import datetime, timedelta
# from celery import Task, group, chain
# from celery.exceptions import SoftTimeLimitExceeded
# from celery_config import app
# from main_video_processor import (
#     get_processor_instance,
#     process_video_request as async_process_video
# )
# import redis

# logger = logging.getLogger(__name__)

# # Redis Cloud client for additional task metadata
# redis_client = redis.Redis(
#     host=os.getenv('REDIS_HOST', 'redis-13640.c251.east-us-mz.azure.redns.redis-cloud.com'),
#     port=int(os.getenv('REDIS_PORT', 13640)),
#     username=os.getenv('REDIS_USERNAME', 'default'),
#     password=os.getenv('REDIS_PASSWORD', 'WgthlYpGR34dVEGlGp8hQgA9fTPMRco2'),
#     decode_responses=True,
#     socket_connect_timeout=5,
#     socket_timeout=5,
#     retry_on_timeout=True,
#     retry_on_error=[redis.ConnectionError, redis.TimeoutError],
#     health_check_interval=30
# )

# # Test Redis connection on module load
# try:
#     redis_client.ping()
#     logger.info("Successfully connected to Redis Cloud")
# except redis.ConnectionError as e:
#     logger.error(f"Failed to connect to Redis Cloud: {e}")
#     logger.warning("Task metadata storage may be unavailable")

# # Rest of the tasks.py code remains the same...

# class VideoProcessingTask(Task):
#     """Base task class with error handling and progress tracking"""
    
#     def on_failure(self, exc, task_id, args, kwargs, einfo):
#         """Handle task failure"""
#         logger.error(f"Task {task_id} failed: {exc}")
#         try:
#             # Update Redis with failure status
#             redis_client.hset(f"task:{task_id}", mapping={
#                 "status": "failed",
#                 "error": str(exc),
#                 "failed_at": datetime.utcnow().isoformat()
#             })
#         except redis.RedisError as e:
#             logger.error(f"Could not update Redis for failed task {task_id}: {e}")
        
#         # Send notification (implement your notification logic)
#         self.send_failure_notification(task_id, exc)
    
#     def on_success(self, retval, task_id, args, kwargs):
#         """Handle task success"""
#         logger.info(f"Task {task_id} completed successfully")
#         try:
#             # Update Redis with success status
#             redis_client.hset(f"task:{task_id}", mapping={
#                 "status": "completed",
#                 "completed_at": datetime.utcnow().isoformat()
#             })
#             redis_client.expire(f"task:{task_id}", 86400)  # Expire after 24 hours
#         except redis.RedisError as e:
#             logger.error(f"Could not update Redis for successful task {task_id}: {e}")
    
#     def on_retry(self, exc, task_id, args, kwargs, einfo):
#         """Handle task retry"""
#         logger.warning(f"Task {task_id} retrying: {exc}")
#         try:
#             redis_client.hset(f"task:{task_id}", mapping={
#                 "status": "retrying",
#                 "retry_reason": str(exc),
#                 "retried_at": datetime.utcnow().isoformat()
#             })
#         except redis.RedisError as e:
#             logger.error(f"Could not update Redis for retrying task {task_id}: {e}")
    
#     def send_failure_notification(self, task_id, exc):
#         """Send notification on task failure (implement based on your needs)"""
#         # Example: Send email, Slack message, etc.
#         pass
# tasks.py
"""
Celery tasks for async video processing
"""
import os
import asyncio
import logging
import tempfile
import shutil
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from celery import Task, group, chain
from celery.exceptions import SoftTimeLimitExceeded
from celery_config import app
from main_video_processor import (
    get_processor_instance,
    process_video_request as async_process_video
)
import redis

logger = logging.getLogger(__name__)

# Redis client for additional task metadata
redis_client = redis.StrictRedis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0)),
    password=os.getenv('REDIS_PASSWORD', None),
    decode_responses=True
)


class VideoProcessingTask(Task):
    """Base task class with error handling and progress tracking"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {exc}")
        # Update Redis with failure status
        redis_client.hset(f"task:{task_id}", mapping={
            "status": "failed",
            "error": str(exc),
            "failed_at": datetime.utcnow().isoformat()
        })
        # Send notification (implement your notification logic)
        self.send_failure_notification(task_id, exc)
    
    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success"""
        logger.info(f"Task {task_id} completed successfully")
        # Update Redis with success status
        redis_client.hset(f"task:{task_id}", mapping={
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat()
        })
        redis_client.expire(f"task:{task_id}", 86400)  # Expire after 24 hours
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry"""
        logger.warning(f"Task {task_id} retrying: {exc}")
        redis_client.hset(f"task:{task_id}", mapping={
            "status": "retrying",
            "retry_reason": str(exc),
            "retried_at": datetime.utcnow().isoformat()
        })
    
    def send_failure_notification(self, task_id, exc):
        """Send notification on task failure (implement based on your needs)"""
        # Example: Send email, Slack message, etc.
        pass

# Include all the task functions from the original tasks.py
# (process_video_task, process_video_high_priority_task, process_batch_task, etc.)
@app.task(
    bind=True,
    base=VideoProcessingTask,
    name='tasks.process_video',
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True
)
def process_video_task(
    self,
    input_video_path: str,
    user_id: str,
    output_video_path: Optional[str] = None,
    keep_intermediate_files: bool = False,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Celery task for processing a single video
    """
    task_id = self.request.id
    logger.info(f"Starting video processing task {task_id} for user {user_id}")
    
    # Store initial task metadata in Redis
    redis_client.hset(f"task:{task_id}", mapping={
        "status": "processing",
        "user_id": user_id,
        "input_video": input_video_path,
        "started_at": datetime.utcnow().isoformat(),
        "progress": "0",
        "current_step": "Initializing"
    })
    
    try:
        # Update progress callback
        def update_progress(step: str, progress: int):
            redis_client.hset(f"task:{task_id}", mapping={
                "current_step": step,
                "progress": str(progress)
            })
            # Send real-time update via WebSocket/SSE if implemented
            self.update_state(
                state='PROGRESS',
                meta={'current': progress, 'total': 100, 'step': step}
            )
        
        # Simulate progress updates (integrate with actual processing)
        update_progress("Extracting audio", 10)
        
        # Run the async video processing in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                async_process_video(
                    input_video_path=input_video_path,
                    user_id=user_id,
                    output_video_path=output_video_path,
                    keep_intermediate_files=keep_intermediate_files,
                    config=config or {}
                )
            )
            
            # Store result metadata
            if result.get('success'):
                redis_client.hset(f"task:{task_id}", mapping={
                    "gdrive_file_id": result.get('gdrive', {}).get('file_id', ''),
                    "gdrive_link": result.get('gdrive', {}).get('web_view_link', ''),
                    "processing_time": str(result.get('processing_time', 0)),
                    "session_id": result.get('session_id', '')
                })
            
            return result
            
        finally:
            loop.close()
            
    except SoftTimeLimitExceeded:
        logger.error(f"Task {task_id} exceeded time limit")
        cleanup_temp_files(output_video_path)
        raise
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        cleanup_temp_files(output_video_path)
        raise


@app.task(
    bind=True,
    base=VideoProcessingTask,
    name='tasks.process_video_high_priority',
    max_retries=3,
    priority=10
)
def process_video_high_priority_task(
    self,
    input_video_path: str,
    user_id: str,
    output_video_path: Optional[str] = None,
    keep_intermediate_files: bool = False,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    High priority video processing task (e.g., for premium users)
    """
    return process_video_task(
        input_video_path,
        user_id,
        output_video_path,
        keep_intermediate_files,
        config
    )


@app.task(
    bind=True,
    name='tasks.process_batch',
    max_retries=2
)
def process_batch_task(
    self,
    video_requests: List[Dict[str, Any]],
    max_concurrent: int = 2
) -> List[str]:
    """
    Process multiple videos as a batch
    Returns list of task IDs for tracking
    """
    task_id = self.request.id
    logger.info(f"Starting batch processing task {task_id} with {len(video_requests)} videos")
    
    # Create a group of subtasks
    job = group(
        process_video_task.s(
            input_video_path=req['input_path'],
            user_id=req['user_id'],
            output_video_path=req.get('output_path'),
            keep_intermediate_files=req.get('keep_intermediate_files', False),
            config=req.get('config', {})
        )
        for req in video_requests
    )
    
    # Execute the group
    result = job.apply_async()
    
    # Store batch metadata
    redis_client.hset(f"batch:{task_id}", mapping={
        "total_videos": str(len(video_requests)),
        "subtask_ids": ",".join([t.id for t in result.results]),
        "started_at": datetime.utcnow().isoformat()
    })
    
    return [t.id for t in result.results]


@app.task(name='tasks.cleanup_old_results')
def cleanup_old_results():
    """
    Periodic task to cleanup old results and temporary files
    """
    logger.info("Starting cleanup of old results")
    
    # Clean up Redis keys older than 24 hours
    pattern = "task:*"
    cursor = 0
    cleaned_count = 0
    
    while True:
        cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
        
        for key in keys:
            task_data = redis_client.hgetall(key)
            if task_data.get('completed_at'):
                completed_at = datetime.fromisoformat(task_data['completed_at'])
                if datetime.utcnow() - completed_at > timedelta(hours=24):
                    redis_client.delete(key)
                    cleaned_count += 1
        
        if cursor == 0:
            break
    
    logger.info(f"Cleaned up {cleaned_count} old task results")
    
    # Clean up temporary files
    temp_dir = tempfile.gettempdir()
    cleanup_temp_directory(temp_dir)
    
    return {"cleaned_tasks": cleaned_count}


@app.task(name='tasks.monitor_health')
def monitor_health():
    """
    Monitor the health of the video processing system
    """
    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "active_tasks": 0,
        "pending_tasks": 0,
        "failed_tasks": 0,
        "completed_tasks": 0,
        "redis_connected": False,
        "workers_active": 0
    }
    
    try:
        # Check Redis connection
        redis_client.ping()
        stats["redis_connected"] = True
        
        # Get task statistics
        inspect = app.control.inspect()
        
        # Get active tasks
        active = inspect.active()
        if active:
            stats["active_tasks"] = sum(len(tasks) for tasks in active.values())
        
        # Get reserved tasks
        reserved = inspect.reserved()
        if reserved:
            stats["pending_tasks"] = sum(len(tasks) for tasks in reserved.values())
        
        # Get worker stats
        stats_data = inspect.stats()
        if stats_data:
            stats["workers_active"] = len(stats_data)
        
        # Count completed and failed tasks from Redis
        for key in redis_client.scan_iter("task:*"):
            task_data = redis_client.hget(key, "status")
            if task_data == "completed":
                stats["completed_tasks"] += 1
            elif task_data == "failed":
                stats["failed_tasks"] += 1
        
        # Store health stats
        redis_client.hset("system:health", mapping=stats)
        redis_client.expire("system:health", 300)  # Expire after 5 minutes
        
        logger.info(f"System health: {stats}")
        
        # Alert if issues detected
        if not stats["redis_connected"] or stats["workers_active"] == 0:
            send_health_alert(stats)
        
    except Exception as e:
        logger.error(f"Health monitoring failed: {e}")
        stats["error"] = str(e)
    
    return stats


@app.task(bind=True, name='tasks.get_task_progress')
def get_task_progress(self, task_id: str) -> Dict[str, Any]:
    """
    Get the current progress of a task
    """
    task_data = redis_client.hgetall(f"task:{task_id}")
    
    if not task_data:
        # Try to get from Celery result backend
        result = app.AsyncResult(task_id)
        
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "info": result.info if not result.ready() else None
        }
    
    return task_data


# Utility functions
def cleanup_temp_files(file_path: Optional[str]):
    """Clean up temporary files on task failure"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up file {file_path}: {e}")


def cleanup_temp_directory(temp_dir: str, age_hours: int = 24):
    """Clean up old temporary files"""
    current_time = time.time()
    age_seconds = age_hours * 3600
    
    for filename in os.listdir(temp_dir):
        if filename.startswith("tmp") or filename.endswith((".mp4", ".wav", ".srt")):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > age_seconds:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")


def send_health_alert(stats: Dict[str, Any]):
    """Send alert when health issues detected"""
    # Implement your alerting logic here
    # Example: Send email, Slack notification, PagerDuty alert, etc.
    logger.error(f"System health alert: {stats}")