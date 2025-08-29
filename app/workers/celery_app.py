# app/workers/celery_app.py
from celery import Celery
from app.config import get_settings
import logging

# Configure logging for Celery
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/celery.log')
    ]
)

logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "video_processor",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.processing_worker", "app.workers.ffmpeg_worker"]
)

# Celery configuration with two different queues
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_reject_on_worker_lost=True,
    task_acks_late=True,
    
    # Results
    result_expires=3600,
    result_persistent=True,
    
    # Task routing - CRITICAL: Different queues for different workers
    task_routes={
        "app.workers.processing_worker.process_video_content": {"queue": "processing"},
        "app.workers.ffmpeg_worker.process_video_with_ffmpeg": {"queue": "ffmpeg"}
    },
    
    # Worker settings
    worker_disable_rate_limits=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    worker_hijack_root_logger=False,
    
    # Time limits
    task_soft_time_limit=1800,  # 30 minutes
    task_time_limit=2400,       # 40 minutes
)
# logger = logging.getLogger(__name__)

# # Get settings
# settings = get_settings()

# # Create Celery app
# celery_app = Celery(
#     "video_processor",
#     broker=settings.celery_broker_url,
#     backend=settings.celery_result_backend,
#     include=["app.workers.video_worker"]
# )

# # Celery configuration
# celery_app.conf.update(
#     # Serialization
#     task_serializer="json",
#     result_serializer="json",
#     accept_content=["json"],
    
#     # Timezone
#     timezone="UTC",
#     enable_utc=True,
    
#     # Task execution
#     task_track_started=True,
#     task_reject_on_worker_lost=True,
#     task_acks_late=True,
#     worker_prefetch_multiplier=1,  # Process one task at a time per worker
    
#     # Result backend settings
#     result_expires=3600,  # Results expire after 1 hour
#     result_persistent=True,
    
#     # Task routing
#     task_routes={
#         "app.workers.video_worker.process_video_with_ffmpeg": {"queue": "video_processing"}
#     },
    
#     # Worker settings
#     worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
#     worker_disable_rate_limits=True,
    
#     # Retry policy
#     task_default_retry_delay=60,  # Retry after 60 seconds
#     task_max_retries=3,
    
#     # Monitoring
#     worker_send_task_events=True,
#     task_send_sent_event=True,
    
#     # Security
#     worker_hijack_root_logger=False,
    
#     # Task time limits
#     task_soft_time_limit=1800,  # 30 minutes soft limit
#     task_time_limit=2400,       # 40 minutes hard limit
# )

# Task failure handler
@celery_app.task(bind=True)
def task_failure_handler(self, task_id, error, traceback):
    """Handle task failures"""
    logger.error(f"Task {task_id} failed: {error}")
    # Could implement notification logic here

# Beat schedule for periodic tasks (if needed)
celery_app.conf.beat_schedule = {
    'cleanup-old-tasks': {
        'task': 'app.workers.maintenance.cleanup_old_tasks',
        'schedule': 24 * 60 * 60.0,  # Daily cleanup
    },
}

# Signal handlers
from celery.signals import worker_ready, worker_shutting_down, task_prerun, task_postrun

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready signal"""
    logger.info(f"Worker {sender} is ready")

@worker_shutting_down.connect  
def worker_shutting_down_handler(sender=None, **kwargs):
    """Handle worker shutdown signal"""
    logger.info(f"Worker {sender} is shutting down")

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task prerun signal"""
    logger.info(f"Task {task.name}[{task_id}] started")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                        retval=None, state=None, **kwds):
    """Handle task postrun signal"""
    logger.info(f"Task {task.name}[{task_id}] finished with state: {state}")

if __name__ == '__main__':
    celery_app.start()