# # celery_config.py
# """
# Celery configuration with Redis Cloud connection
# """
# import os
# from celery import Celery
# from kombu import Queue, Exchange
# from datetime import timedelta

# # Redis Cloud configuration
# REDIS_HOST = os.getenv('REDIS_HOST', 'redis-13640.c251.east-us-mz.azure.redns.redis-cloud.com')
# REDIS_PORT = int(os.getenv('REDIS_PORT', 13640))
# REDIS_USERNAME = os.getenv('REDIS_USERNAME', 'default')
# REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', 'WgthlYpGR34dVEGlGp8hQgA9fTPMRco2')

# # Build Redis URL for Celery (format: redis://username:password@host:port/db)
# REDIS_URL = f'redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0'

# # Initialize Celery
# app = Celery('video_processor')

# # Celery configuration
# app.conf.update(
#     broker_url=REDIS_URL,
#     result_backend=REDIS_URL,

#     broker_pool_limit=1,
    
#     # Redis-specific settings for cloud connection
#     broker_connection_retry_on_startup=True,
#     broker_connection_retry=True,
#     broker_connection_max_retries=10,
    
#     # Task execution settings
#     task_serializer='json',
#     accept_content=['json'],
#     result_serializer='json',
#     timezone='UTC',
#     enable_utc=True,
    
#     # Task routing
#     task_routes={
#         'tasks.process_video': {'queue': 'video_processing'},
#         'tasks.process_video_high_priority': {'queue': 'high_priority'},
#         'tasks.process_batch': {'queue': 'batch_processing'},
#     },
    
#     # Queue configuration
#     task_queues=(
#         Queue('video_processing', Exchange('video_processing'), routing_key='video_processing'),
#         Queue('high_priority', Exchange('high_priority'), routing_key='high_priority', priority=10),
#         Queue('batch_processing', Exchange('batch_processing'), routing_key='batch_processing'),
#     ),
    
#     # Worker settings
#     worker_prefetch_multiplier=1,
#     worker_max_tasks_per_child=10,
    
#     # Task time limits
#     task_soft_time_limit=1800,  # 30 minutes soft limit
#     task_time_limit=3600,  # 60 minutes hard limit
    
#     # Result backend settings
#     result_expires=86400,  # Results expire after 24 hours
#     result_compression='gzip',
    
#     # Retry settings
#     task_acks_late=True,
#     task_reject_on_worker_lost=True,
    
#     # Monitoring
#     worker_send_task_events=True,
#     task_send_sent_event=True,
# )

# # Beat schedule for periodic tasks
# app.conf.beat_schedule = {
#     'cleanup-old-results': {
#         'task': 'tasks.cleanup_old_results',
#         'schedule': timedelta(hours=6),
#     },
#     'monitor-processing-health': {
#         'task': 'tasks.monitor_health',
#         'schedule': timedelta(minutes=5),
#     },
# }

# celery_config.py
"""
Celery configuration and task definitions for video processing
"""
import os
from celery import Celery
from kombu import Queue, Exchange
from datetime import timedelta

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

# Build Redis URL
if REDIS_PASSWORD:
    REDIS_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
else:
    REDIS_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

# Initialize Celery
app = Celery('video_processor')

# Celery configuration
app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'tasks.process_video': {'queue': 'video_processing'},
        'tasks.process_video_high_priority': {'queue': 'high_priority'},
        'tasks.process_batch': {'queue': 'batch_processing'},
    },
    
    # Queue configuration
    task_queues=(
        Queue('video_processing', Exchange('video_processing'), routing_key='video_processing'),
        Queue('high_priority', Exchange('high_priority'), routing_key='high_priority', priority=10),
        Queue('batch_processing', Exchange('batch_processing'), routing_key='batch_processing'),
    ),
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Only fetch one task at a time for long-running tasks
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks to prevent memory leaks
    
    # Task time limits
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=3600,  # 60 minutes hard limit
    
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    result_compression='gzip',
    
    # Retry settings
    task_acks_late=True,  # Tasks acknowledged after completion
    task_reject_on_worker_lost=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Beat schedule for periodic tasks (optional)
app.conf.beat_schedule = {
    'cleanup-old-results': {
        'task': 'tasks.cleanup_old_results',
        'schedule': timedelta(hours=6),
    },
    'monitor-processing-health': {
        'task': 'tasks.monitor_health',
        'schedule': timedelta(minutes=5),
    },
}