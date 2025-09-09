# # app/main.py
# from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional, Dict, Any
# import os
# import uuid
# import tempfile
# import shutil
# from pathlib import Path
# import logging
# import asyncio
# from datetime import datetime
# import json

# from app.models import TaskResponse, VideoProcessingRequest, TaskStatus
# from app.services.video_processor import AsyncVideoProcessor
# from app.workers.celery_app import celery_app
# from app.database.mongodb_manager import MongoDBManager
# from app.config import get_settings
# from app.middleware.security import RateLimitMiddleware, SecurityHeadersMiddleware

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('logs/app.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(
#     title="Video Processing API",
#     description="Production async video transcription and optimization service",
#     version="2.0.0",
#     docs_url="/docs" if get_settings().environment != "production" else None,
#     redoc_url="/redoc" if get_settings().environment != "production" else None
# )

# # Security middleware
# app.add_middleware(SecurityHeadersMiddleware)
# app.add_middleware(RateLimitMiddleware, calls=50, period=60)  # 50 requests per minute

# # CORS middleware - configure restrictively for production
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with your frontend domains
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "DELETE"],
#     allow_headers=["*"],
# )

# # Global instances
# settings = get_settings()
# mongodb_manager = None

# # File size limit (500MB)
# MAX_FILE_SIZE = 500 * 1024 * 1024

# @app.on_event("startup")
# async def startup_event():
#     """Initialize database connections and create directories"""
#     global mongodb_manager
    
#     # Create necessary directories
#     os.makedirs("uploads", exist_ok=True)
#     os.makedirs("logs", exist_ok=True)
    
#     # Initialize MongoDB
#     mongodb_manager = MongoDBManager(
#         mongodb_uri=settings.mongodb_uri,
#         database_name=settings.mongodb_database
#     )
#     await mongodb_manager.initialize()
    
#     logger.info("Video Processing API started successfully")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     global mongodb_manager
#     if mongodb_manager:
#         await mongodb_manager.close()
#     logger.info("Application shutdown complete")

# @app.get("/health")
# async def health_check():
#     """Health check endpoint with service status"""
#     try:
#         # Check MongoDB connection
#         await mongodb_manager.client.admin.command('ping')
#         mongo_status = "healthy"
#     except:
#         mongo_status = "unhealthy"
    
#     # Check Celery workers
#     inspect = celery_app.control.inspect()
#     workers = inspect.active()
#     worker_count = len(workers) if workers else 0
    
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow().isoformat(),
#         "services": {
#             "mongodb": mongo_status,
#             "celery_workers": worker_count
#         }
#     }

# @app.post("/process-video", response_model=TaskResponse)
# async def process_video(
#     user_id: str,
#     file: UploadFile = File(...),
#     background_image_path: Optional[str] = None,
#     overlay_options: Optional[str] = None  # JSON string
# ):
#     """Submit video for processing with enhanced validation"""
    
#     # Validate file type
#     allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
#     file_extension = Path(file.filename).suffix.lower()
#     if file_extension not in allowed_extensions:
#         raise HTTPException(
#             status_code=400, 
#             detail=f"Unsupported format. Allowed: {', '.join(allowed_extensions)}"
#         )
    
#     # Check file size
#     if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
#         raise HTTPException(
#             status_code=413, 
#             detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
#         )
    
#     # Generate unique task ID
#     task_id = str(uuid.uuid4())
    
#     # Create upload directory
#     upload_dir = f"uploads/{task_id}"
#     os.makedirs(upload_dir, exist_ok=True)
#     input_path = os.path.join(upload_dir, f"input_{file.filename}")
    
#     try:
#         # Save uploaded file
#         with open(input_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         file_size = os.path.getsize(input_path)
#         logger.info(f"File uploaded: {file.filename} ({file_size} bytes)")
        
#         # Parse overlay options if provided
#         config = {}
#         if background_image_path:
#             config["background_image_path"] = background_image_path
        
#         if overlay_options:
#             try:
#                 config["overlay_options"] = json.loads(overlay_options)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid overlay_options JSON")
        
#         # Create initial task record
#         initial_task_data = {
#             "task_id": task_id,
#             "user_id": user_id,
#             "filename": file.filename,
#             "file_size": file_size,
#             "status": "received",
#             "created_at": datetime.utcnow(),
#             "input_path": input_path,
#             "upload_dir": upload_dir,
#             "config": config
#         }
        
#         await mongodb_manager.create_task_record(initial_task_data)
        
#         # Start async processing
#         await start_video_processing_async(task_id, user_id, input_path, upload_dir, config)
        
#         return TaskResponse(
#             task_id=task_id,
#             status="processing",
#             message="Video processing started successfully"
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         # Cleanup on error
#         if os.path.exists(upload_dir):
#             shutil.rmtree(upload_dir, ignore_errors=True)
        
#         logger.error(f"Error processing upload: {e}")
#         raise HTTPException(status_code=500, detail="Internal processing error")

# async def start_video_processing_async(
#     task_id: str, 
#     user_id: str, 
#     input_path: str, 
#     upload_dir: str, 
#     config: Dict[str, Any]
# ):
#     """Async processing pipeline - non-FFmpeg operations only"""
#     try:
#         # Initialize processor
#         processor = AsyncVideoProcessor(config)
        
#         # Step 1: Transcription
#         await mongodb_manager.update_task_status(task_id, "transcribing")
#         logger.info(f"Starting transcription for task {task_id}")
        
#         language, srt_content = await processor.transcriber.transcribe_video_async(input_path)
        
#         if not srt_content or not srt_content.strip():
#             raise ValueError("Transcription failed - no content generated")
        
#         await mongodb_manager.update_task_data(task_id, {
#             "transcription": {
#                 "language": language,
#                 "content_length": len(srt_content)
#             }
#         })
        
#         # Step 2: AI Optimization  
#         await mongodb_manager.update_task_status(task_id, "optimizing")
        
#         segments = processor.summarizer.parse_srt_content(srt_content)
#         optimized_segments = await processor.summarizer.optimize_all_segments_async(segments)
        
#         # Step 3: Audio Generation (Non-FFmpeg)
#         await mongodb_manager.update_task_status(task_id, "generating_audio")
        
#         segment_audio_data = []
#         audio_tasks = []
        
#         # Generate audio for each segment concurrently
#         for i, opt_seg in enumerate(optimized_segments):
#             segment_dir = os.path.join(upload_dir, f"segment_{i}")
#             os.makedirs(segment_dir, exist_ok=True)
#             audio_tasks.append(
#                 processor.summarizer.generate_tts_audio_async(opt_seg.optimized_text, segment_dir)
#             )
        
#         audio_results = await asyncio.gather(*audio_tasks, return_exceptions=True)
        
#         # Process audio generation results
#         for i, result in enumerate(audio_results):
#             if isinstance(result, Exception) or not result or not result[0]:
#                 logger.warning(f"Audio generation failed for segment {i}")
#                 continue
            
#             audio_path, audio_duration = result
#             segment_audio_data.append({
#                 "index": i,
#                 "audio_path": audio_path,
#                 "audio_duration": audio_duration,
#                 "optimized_segment": serialize_optimized_segment(optimized_segments[i])
#             })
        
#         if not segment_audio_data:
#             raise ValueError("All audio generation failed")
        
#         logger.info(f"Generated audio for {len(segment_audio_data)}/{len(optimized_segments)} segments")
        
#         # Step 4: Queue FFmpeg processing
#         await mongodb_manager.update_task_status(task_id, "queued_for_video_processing")
        
#         celery_task = celery_app.send_task(
#             'app.workers.video_worker.process_video_with_ffmpeg',
#             args=[task_id, input_path, segment_audio_data, config],
#             task_id=f"ffmpeg_{task_id}"
#         )
        
#         await mongodb_manager.update_task_data(task_id, {"celery_task_id": celery_task.id})
#         logger.info(f"Task {task_id} queued for FFmpeg processing")
        
#     except Exception as e:
#         logger.error(f"Processing error for task {task_id}: {e}")
#         await mongodb_manager.update_task_status(task_id, "failed", error=str(e))
        
#         # Cleanup
#         if os.path.exists(upload_dir):
#             shutil.rmtree(upload_dir, ignore_errors=True)

# def serialize_optimized_segment(opt_seg):
#     """Serialize OptimizedSegment for Celery"""
#     return {
#         "original": {
#             "index": opt_seg.original.index,
#             "start_time": opt_seg.original.start_time,
#             "end_time": opt_seg.original.end_time,
#             "start_seconds": opt_seg.original.start_seconds,
#             "end_seconds": opt_seg.original.end_seconds,
#             "duration": opt_seg.original.duration,
#             "text": opt_seg.original.text,
#             "word_count": opt_seg.original.word_count,
#             "words_per_second": opt_seg.original.words_per_second
#         },
#         "optimized_text": opt_seg.optimized_text,
#         "optimized_word_count": opt_seg.optimized_word_count,
#         "estimated_speech_duration": opt_seg.estimated_speech_duration,
#         "speed_multiplier": opt_seg.speed_multiplier,
#         "reasoning": opt_seg.reasoning
#     }

# @app.get("/task/{task_id}/status", response_model=TaskStatus)
# async def get_task_status(task_id: str):
#     """Get comprehensive task status"""
#     task_data = await mongodb_manager.get_task_by_id(task_id)
    
#     if not task_data:
#         raise HTTPException(status_code=404, detail="Task not found")
    
#     # Add Celery task info if available
#     if task_data.get("celery_task_id"):
#         celery_result = celery_app.AsyncResult(task_data["celery_task_id"])
#         task_data["celery_status"] = celery_result.status
#         if hasattr(celery_result, 'info') and celery_result.info:
#             task_data["progress"] = celery_result.info.get("progress")
    
#     return TaskStatus(**task_data)

# @app.get("/user/{user_id}/tasks")
# async def get_user_tasks(user_id: str, limit: int = 20, offset: int = 0):
#     """Get user's tasks with pagination"""
#     if limit > 100:
#         limit = 100  # Prevent excessive queries
        
#     tasks = await mongodb_manager.get_user_tasks(user_id, limit, offset)
#     total_count = await mongodb_manager.count_user_tasks(user_id)
    
#     return {
#         "tasks": tasks,
#         "total": total_count,
#         "limit": limit,
#         "offset": offset
#     }

# @app.delete("/task/{task_id}")
# async def cancel_task(task_id: str, user_id: str):
#     """Cancel task with user verification"""
#     task_data = await mongodb_manager.get_task_by_id(task_id)
    
#     if not task_data:
#         raise HTTPException(status_code=404, detail="Task not found")
    
#     if task_data["user_id"] != user_id:
#         raise HTTPException(status_code=403, detail="Unauthorized")
    
#     # Cancel Celery task if exists
#     if task_data.get("celery_task_id"):
#         celery_app.control.revoke(task_data["celery_task_id"], terminate=True)
    
#     await mongodb_manager.update_task_status(task_id, "cancelled")
    
#     # Cleanup files
#     if task_data.get("upload_dir") and os.path.exists(task_data["upload_dir"]):
#         shutil.rmtree(task_data["upload_dir"], ignore_errors=True)
    
#     return {"message": "Task cancelled successfully"}

# @app.get("/stats")
# async def get_system_stats():
#     """System statistics endpoint"""
#     return await mongodb_manager.get_processing_stats()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app.main:app", host="localhost", port=8000, reload=False)

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
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
from app.auth.routes import router as auth_router
from app.auth.middleware import get_current_user
from app.auth.service import AuthService
from app.auth.routes import router as auth_router
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
app.include_router(auth_router)
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
    file: UploadFile = File(...),
    background_image_path: Optional[str] = None,
    overlay_options: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Submit video for processing - requires authentication"""
    
    # Get user_id from authenticated user instead of parameter
    user_id = str(current_user["_id"])
    
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
        logger.info(f"File uploaded by user {user_id}: {file.filename} ({file_size} bytes)")
        
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
        
        logger.info(f"Task {task_id} submitted to processing worker by user {user_id}: {processing_task.id}")
        
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
        
        logger.error(f"Error processing upload for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/task/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """Get comprehensive task status"""
    task_data = await mongodb_manager.get_task_by_id(task_id)
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # --- START OF FIX ---

    # Check Celery task for the processing worker
    if task_data.get("processing_task_id"):
        processing_result = celery_app.AsyncResult(task_data["processing_task_id"])
        task_data["processing_status"] = processing_result.status
        
        # This is the corrected logic.
        # We check if the task was successful before trying to access the result.
        if processing_result.successful():
            # On success, .info should be the dictionary returned by your task
            if isinstance(processing_result.info, dict):
                task_data["processing_progress"] = processing_result.info.get("progress")
        elif processing_result.failed():
            # On failure, .info holds the exception. We convert it to a string.
            task_data["error"] = str(processing_result.info)

    # Check Celery task for the ffmpeg worker
    if task_data.get("ffmpeg_task_id"):
        ffmpeg_result = celery_app.AsyncResult(task_data["ffmpeg_task_id"])
        task_data["ffmpeg_status"] = ffmpeg_result.status
        
        # Applying the same corrected logic here.
        if ffmpeg_result.successful():
            # Safely access the progress dictionary only on success.
            if isinstance(ffmpeg_result.info, dict):
                task_data["ffmpeg_progress"] = ffmpeg_result.info.get("progress")
        elif ffmpeg_result.failed():
            # If the task failed, capture the error message.
            # This will prevent the AttributeError.
            task_data["error"] = str(ffmpeg_result.info)

    # --- END OF FIX ---
    
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

@app.get("/videos")
async def get_user_videos(
    limit: int = 50, 
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get all videos for the current user with complete details including download links"""
    if limit > 100:
        limit = 100
    
    user_id = str(current_user["_id"])
    
    try:
        # Get all tasks for the user (both completed and processing)
        all_tasks = await mongodb_manager.get_user_tasks(user_id, 1000, 0)  # Get more to filter properly
        
        # Get legacy video records
        legacy_videos = await mongodb_manager.get_user_videos(user_id, 1000, 0)
        
        # Process tasks into video objects
        video_objects = []
        
        for task in all_tasks:
            video_obj = {
                "id": task.get("task_id"),
                "user_id": task.get("user_id"),
                "filename": task.get("filename"),
                "original_filename": task.get("filename"),
                "status": task.get("status"),
                "created_at": task.get("created_at"),
                "updated_at": task.get("updated_at"),
                "file_size": task.get("file_size"),
                "progress": task.get("progress"),
                "error": task.get("error"),
                "type": "processed_video"
            }
            
            # Add result data if available
            if task.get("result"):
                result = task["result"]
                video_obj.update({
                    "processing_time": result.get("processing_time"),
                    "cost_summary": result.get("cost_summary"),
                    "video_duration": result.get("video_duration"),
                    "optimization_stats": result.get("optimization_stats"),
                    "completed_at": result.get("completed_at")
                })
                
                # Extract Google Drive data from result.gdrive_data
                if result.get("gdrive_data"):
                    gdrive_data = result["gdrive_data"]
                    video_obj.update({
                        "output_filename": gdrive_data.get("file_name"),
                        "gdrive_file_id": gdrive_data.get("file_id"),
                        "gdrive_share_url": gdrive_data.get("web_view_link"),
                        "download_url": gdrive_data.get("download_link"),
                        "gdrive_download_url": gdrive_data.get("download_link"),
                        "output_file_size": gdrive_data.get("size")
                    })
            
            # Add transcription data if available
            if task.get("transcription"):
                video_obj["transcription"] = task["transcription"]
            
            # Add optimization data if available
            if task.get("optimization"):
                video_obj["optimization"] = task["optimization"]
            
            # Legacy: Check for gdrive_data at task level (for backwards compatibility)
            if task.get("gdrive_data") and not video_obj.get("gdrive_file_id"):
                gdrive_data = task["gdrive_data"]
                video_obj.update({
                    "gdrive_file_id": gdrive_data.get("file_id"),
                    "gdrive_share_url": gdrive_data.get("share_url") or gdrive_data.get("web_view_link"),
                    "gdrive_download_url": gdrive_data.get("download_url") or gdrive_data.get("download_link"),
                    "download_url": gdrive_data.get("download_url") or gdrive_data.get("download_link")
                })
            
            video_objects.append(video_obj)
        
        # Process legacy videos
        for video in legacy_videos:
            video_obj = {
                "id": video.get("session_id"),
                "user_id": video.get("user_id"),
                "filename": video.get("gdrive_data", {}).get("filename", "Unknown"),
                "original_filename": video.get("gdrive_data", {}).get("filename", "Unknown"),
                "status": "completed",
                "created_at": video.get("created_at"),
                "updated_at": video.get("created_at"),
                "type": "legacy_video"
            }
            
            # Add gdrive data
            if video.get("gdrive_data"):
                gdrive_data = video["gdrive_data"]
                video_obj.update({
                    "gdrive_file_id": gdrive_data.get("file_id"),
                    "gdrive_share_url": gdrive_data.get("share_url") or gdrive_data.get("web_view_link"),
                    "gdrive_download_url": gdrive_data.get("download_url") or gdrive_data.get("download_link"),
                    "download_url": gdrive_data.get("download_url") or gdrive_data.get("download_link"),
                    "output_filename": gdrive_data.get("file_name") or gdrive_data.get("filename")
                })
            
            # Add processing data
            if video.get("processing_data"):
                processing_data = video["processing_data"]
                video_obj.update({
                    "processing_time": processing_data.get("processing_time"),
                    "video_duration": processing_data.get("video_duration"),
                    "optimization_stats": processing_data.get("optimization_stats")
                })
            
            video_objects.append(video_obj)
        
        # Sort by creation date (newest first)
        video_objects.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
        
        # Apply pagination
        total_count = len(video_objects)
        paginated_videos = video_objects[offset:offset + limit]
        
        return {
            "videos": paginated_videos,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_count
        }
        
    except Exception as e:
        logger.error(f"Error getting videos for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve videos")

@app.get("/video/{task_id}")
async def get_video_details(task_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get details of a single processed video by task_id.
    Only accessible by the owner of the video.
    """
    task_data = await mongodb_manager.get_task_by_id(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Video not found")
    if str(task_data.get("user_id")) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Build comprehensive video details
    video_details = {
        "id": task_data.get("task_id"),
        "user_id": task_data.get("user_id"),
        "filename": task_data.get("filename"),
        "original_filename": task_data.get("filename"),
        "status": task_data.get("status"),
        "created_at": task_data.get("created_at"),
        "updated_at": task_data.get("updated_at"),
        "file_size": task_data.get("file_size"),
        "error": task_data.get("error"),
        "progress": task_data.get("progress"),
        "input_path": task_data.get("input_path"),
        "upload_dir": task_data.get("upload_dir"),
        "config": task_data.get("config"),
        "processing_task_id": task_data.get("processing_task_id"),
        "ffmpeg_task_id": task_data.get("ffmpeg_task_id"),
        "transcription": task_data.get("transcription"),
        "optimization": task_data.get("optimization"),
        "type": "processed_video"
    }
    
    # Add result data if available
    if task_data.get("result"):
        result = task_data["result"]
        video_details.update({
            "result": result,  # Keep full result for backwards compatibility
            "processing_time": result.get("processing_time"),
            "cost_summary": result.get("cost_summary"),
            "video_duration": result.get("video_duration"),
            "optimization_stats": result.get("optimization_stats"),
            "completed_at": result.get("completed_at"),
            "processing_details": result.get("processing_details")
        })
        
        # Extract and flatten Google Drive data
        if result.get("gdrive_data"):
            gdrive_data = result["gdrive_data"]
            video_details.update({
                "gdrive_data": gdrive_data,  # Keep nested structure for backwards compatibility
                "output_filename": gdrive_data.get("file_name"),
                "gdrive_file_id": gdrive_data.get("file_id"),
                "gdrive_share_url": gdrive_data.get("web_view_link"),
                "download_url": gdrive_data.get("download_link"),
                "gdrive_download_url": gdrive_data.get("download_link"),
                "output_file_size": gdrive_data.get("size"),
                "gdrive_success": gdrive_data.get("success")
            })
    
    # Legacy: Check for gdrive_data at task level (for backwards compatibility)
    if task_data.get("gdrive_data") and not video_details.get("gdrive_file_id"):
        gdrive_data = task_data["gdrive_data"]
        video_details.update({
            "gdrive_data": gdrive_data,
            "gdrive_file_id": gdrive_data.get("file_id"),
            "gdrive_share_url": gdrive_data.get("share_url") or gdrive_data.get("web_view_link"),
            "gdrive_download_url": gdrive_data.get("download_url") or gdrive_data.get("download_link"),
            "download_url": gdrive_data.get("download_url") or gdrive_data.get("download_link")
        })
    
    return video_details

@app.get("/stats")
async def get_system_stats(current_user: dict = Depends(get_current_user)):
    """System statistics endpoint - returns stats for current user"""
    user_id = str(current_user["_id"])
    return await mongodb_manager.get_processing_stats(user_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=False)
 

