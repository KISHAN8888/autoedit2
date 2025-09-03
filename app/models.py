# app/models.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class VideoProcessingRequest(BaseModel):
    user_id: str
    config: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatus(BaseModel):
    task_id: str
    user_id: str
    filename: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    progress: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    gdrive_data: Optional[Dict[str, Any]] = None
    transcription: Optional[Dict[str, Any]] = None
    optimization: Optional[Dict[str, Any]] = None
    audio_generation: Optional[Dict[str, Any]] = None
    celery_task_id: Optional[str] = None
    celery_status: Optional[str] = None
    file_size: Optional[int] = None

class ProcessingStats(BaseModel):
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    average_processing_time: Optional[float] = None
    total_cost: Optional[float] = None

class UserTasksResponse(BaseModel):
    tasks: List[TaskStatus]
    total: int
    limit: int
    offset: int