# from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
# from pydantic import BaseModel
# import asyncio
# import uuid
# import os
# import tempfile
# import shutil
# from typing import Optional, Dict, Any
# from main_video_processor import process_video_request
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(
#     title="Async Video Processing API",
#     description="High-performance async video transcription and optimization service",
#     version="1.0.0"
# )

# class ProcessingRequest(BaseModel):
#     input_video_path: str
#     output_video_path: Optional[str] = None
#     keep_intermediate_files: bool = False
#     config: Dict[str, Any] = {}

# class ProcessingStatus(BaseModel):
#     task_id: str
#     status: str
#     result: Optional[Dict[str, Any]] = None
#     error: Optional[str] = None
#     progress: Optional[str] = None

# class UploadRequest(BaseModel):
#     output_video_path: Optional[str] = None
#     keep_intermediate_files: bool = False
#     config: Dict[str, Any] = {}

# # In-memory task storage (use Redis/Database in production)
# tasks: Dict[str, Dict] = {}

# @app.on_event("startup")
# async def startup_event():
#     """Initialize the application on startup"""
#     logger.info("Starting Async Video Processing API")
#     # Warm up the processor instance
#     try:
#         from main_video_processor import get_processor_instance
#         await get_processor_instance()
#         logger.info("Processor instance initialized successfully")
#     except Exception as e:
#         logger.error(f"Failed to initialize processor: {e}")

# @app.get("/")
# async def root():
#     """Health check endpoint"""
#     return {
#         "message": "Async Video Processing API",
#         "status": "running",
#         "endpoints": {
#             "process": "/process-video/",
#             "upload": "/upload-and-process/",
#             "status": "/status/{task_id}",
#             "tasks": "/tasks/"
#         }
#     }

# @app.post("/process-video/")
# async def process_video_endpoint(request: ProcessingRequest, background_tasks: BackgroundTasks):
#     """
#     Process a video file that exists on the server filesystem
#     """
#     if not os.path.exists(request.input_video_path):
#         raise HTTPException(status_code=404, detail="Input video file not found")
    
#     task_id = str(uuid.uuid4())
#     tasks[task_id] = {
#         "status": "pending",
#         "result": None,
#         "error": None,
#         "progress": "Task queued for processing"
#     }
    
#     # Start background processing
#     background_tasks.add_task(
#         process_video_background, 
#         task_id, 
#         request.input_video_path,
#         request.output_video_path,
#         request.keep_intermediate_files,
#         request.config
#     )
    
#     logger.info(f"Started processing task {task_id} for video: {request.input_video_path}")
#     return {"task_id": task_id, "status": "started"}

# @app.post("/upload-and-process/")
# async def upload_and_process_video(
#     video_file: UploadFile = File(...),
#     background_tasks: BackgroundTasks = None,
#     output_video_path: Optional[str] = None,
#     keep_intermediate_files: bool = False,
#     config: str = "{}"  # JSON string for config
# ):
#     """
#     Upload a video file and process it
#     """
#     import json
    
#     # Parse config JSON
#     try:
#         config_dict = json.loads(config) if config else {}
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=400, detail="Invalid JSON in config parameter")
    
#     # Validate file type
#     if not video_file.content_type.startswith('video/'):
#         raise HTTPException(status_code=400, detail="File must be a video")
    
#     task_id = str(uuid.uuid4())
#     tasks[task_id] = {
#         "status": "uploading",
#         "result": None,
#         "error": None,
#         "progress": "Uploading video file"
#     }
    
#     # Save uploaded file to temporary location
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
#         temp_input_path = temp_file.name
#         shutil.copyfileobj(video_file.file, temp_file)
    
#     logger.info(f"Uploaded video saved to: {temp_input_path}")
    
#     # Start background processing
#     background_tasks.add_task(
#         process_uploaded_video_background,
#         task_id,
#         temp_input_path,
#         output_video_path,
#         keep_intermediate_files,
#         config_dict
#     )
    
#     return {"task_id": task_id, "status": "started", "filename": video_file.filename}

# async def process_video_background(
#     task_id: str, 
#     input_video_path: str, 
#     output_video_path: Optional[str],
#     keep_intermediate_files: bool,
#     config: Dict[str, Any]
# ):
#     """Background task for processing video files"""
#     try:
#         tasks[task_id]["status"] = "processing"
#         tasks[task_id]["progress"] = "Starting video transcription"
        
#         result = await process_video_request(
#             input_video_path=input_video_path,
#             output_video_path=output_video_path,
#             keep_intermediate_files=keep_intermediate_files,
#             config=config
#         )
        
#         if result["success"]:
#             tasks[task_id]["status"] = "completed"
#             tasks[task_id]["result"] = result
#             tasks[task_id]["progress"] = "Processing completed successfully"
#             logger.info(f"Task {task_id} completed successfully")
#         else:
#             tasks[task_id]["status"] = "failed"
#             tasks[task_id]["error"] = result.get("error", "Unknown error")
#             tasks[task_id]["progress"] = "Processing failed"
#             logger.error(f"Task {task_id} failed: {result.get('error')}")
            
#     except Exception as e:
#         tasks[task_id]["status"] = "failed"
#         tasks[task_id]["error"] = str(e)
#         tasks[task_id]["progress"] = "Processing failed with exception"
#         logger.error(f"Task {task_id} failed with exception: {e}")

# async def process_uploaded_video_background(
#     task_id: str,
#     temp_input_path: str,
#     output_video_path: Optional[str],
#     keep_intermediate_files: bool,
#     config: Dict[str, Any]
# ):
#     """Background task for processing uploaded video files"""
#     try:
#         await process_video_background(
#             task_id, 
#             temp_input_path, 
#             output_video_path, 
#             keep_intermediate_files, 
#             config
#         )
#     finally:
#         # Clean up uploaded temporary file
#         try:
#             if os.path.exists(temp_input_path):
#                 os.unlink(temp_input_path)
#                 logger.info(f"Cleaned up uploaded file: {temp_input_path}")
#         except Exception as e:
#             logger.warning(f"Could not clean up uploaded file {temp_input_path}: {e}")

# @app.get("/status/{task_id}")
# async def get_status(task_id: str):
#     """Get the status of a processing task"""
#     if task_id not in tasks:
#         raise HTTPException(status_code=404, detail="Task not found")
    
#     return ProcessingStatus(
#         task_id=task_id,
#         status=tasks[task_id]["status"],
#         result=tasks[task_id]["result"],
#         error=tasks[task_id]["error"],
#         progress=tasks[task_id]["progress"]
#     )

# @app.get("/tasks/")
# async def list_tasks():
#     """List all tasks and their statuses"""
#     return {
#         "total_tasks": len(tasks),
#         "tasks": {
#             task_id: {
#                 "status": task_data["status"],
#                 "progress": task_data["progress"]
#             }
#             for task_id, task_data in tasks.items()
#         }
#     }

# @app.delete("/tasks/{task_id}")
# async def delete_task(task_id: str):
#     """Delete a completed or failed task from memory"""
#     if task_id not in tasks:
#         raise HTTPException(status_code=404, detail="Task not found")
    
#     task_status = tasks[task_id]["status"]
#     if task_status in ["processing", "pending", "uploading"]:
#         raise HTTPException(status_code=400, detail="Cannot delete active task")
    
#     del tasks[task_id]
#     return {"message": f"Task {task_id} deleted successfully"}

# @app.post("/process-multiple/")
# async def process_multiple_videos(
#     requests: list[ProcessingRequest],
#     background_tasks: BackgroundTasks,
#     max_concurrent: int = 2
# ):
#     """
#     Process multiple videos concurrently
#     """
#     if len(requests) > 10:  # Limit batch size
#         raise HTTPException(status_code=400, detail="Maximum 10 videos per batch")
    
#     task_ids = []
    
#     for request in requests:
#         if not os.path.exists(request.input_video_path):
#             raise HTTPException(
#                 status_code=404, 
#                 detail=f"Input video file not found: {request.input_video_path}"
#             )
        
#         task_id = str(uuid.uuid4())
#         task_ids.append(task_id)
#         tasks[task_id] = {
#             "status": "pending",
#             "result": None,
#             "error": None,
#             "progress": "Queued for batch processing"
#         }
    
#     # Start batch processing
#     background_tasks.add_task(
#         process_batch_background,
#         task_ids,
#         requests,
#         max_concurrent
#     )
    
#     return {
#         "message": f"Started batch processing of {len(requests)} videos",
#         "task_ids": task_ids,
#         "max_concurrent": max_concurrent
#     }

# async def process_batch_background(
#     task_ids: list[str],
#     requests: list[ProcessingRequest],
#     max_concurrent: int
# ):
#     """Background task for batch processing"""
#     from main_video_processor import process_multiple_videos_concurrently
    
#     # Convert requests to the format expected by process_multiple_videos_concurrently
#     video_requests = []
#     for i, request in enumerate(requests):
#         video_requests.append({
#             'input_path': request.input_video_path,
#             'output_path': request.output_video_path,
#             'keep_intermediate_files': request.keep_intermediate_files,
#             'config': request.config
#         })
#         tasks[task_ids[i]]["status"] = "processing"
#         tasks[task_ids[i]]["progress"] = "Processing in batch"
    
#     try:
#         results = await process_multiple_videos_concurrently(video_requests, max_concurrent)
        
#         # Update task statuses with results
#         for i, (task_id, result) in enumerate(zip(task_ids, results)):
#             if isinstance(result, Exception):
#                 tasks[task_id]["status"] = "failed"
#                 tasks[task_id]["error"] = str(result)
#                 tasks[task_id]["progress"] = "Batch processing failed"
#             elif result.get('success'):
#                 tasks[task_id]["status"] = "completed"
#                 tasks[task_id]["result"] = result
#                 tasks[task_id]["progress"] = "Batch processing completed"
#             else:
#                 tasks[task_id]["status"] = "failed"
#                 tasks[task_id]["error"] = result.get("error", "Unknown error")
#                 tasks[task_id]["progress"] = "Batch processing failed"
                
#     except Exception as e:
#         # Mark all tasks as failed
#         for task_id in task_ids:
#             tasks[task_id]["status"] = "failed"
#             tasks[task_id]["error"] = f"Batch processing error: {str(e)}"
#             tasks[task_id]["progress"] = "Batch processing failed"

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "server:app",
#         host="localhost",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )

# # Run the server with:
# # uvicorn server:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Query, Form
from pydantic import BaseModel
import asyncio
import uuid
import os
import tempfile
import shutil
from typing import Optional, Dict, Any
from main_video_processor import (
    process_video_request, 
    get_user_videos, 
    get_video_by_session, 
    get_processing_stats
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Async Video Processing API with Cloud Storage",
    description="High-performance async video transcription, optimization, and cloud storage service",
    version="2.0.0"
)

class ProcessingRequest(BaseModel):
    input_video_path: str
    user_id: str
    output_video_path: Optional[str] = None
    keep_intermediate_files: bool = False
    config: Dict[str, Any] = {}

class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[str] = None

class UploadRequest(BaseModel):
    user_id: str
    output_video_path: Optional[str] = None
    keep_intermediate_files: bool = False
    config: Dict[str, Any] = {}

# In-memory task storage (use Redis/Database in production)
tasks: Dict[str, Dict] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Async Video Processing API with Cloud Storage")
    # Warm up the processor instance
    try:
        from main_video_processor import get_processor_instance
        await get_processor_instance()
        logger.info("Processor instance initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Async Video Processing API with Cloud Storage",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Video transcription and optimization",
            "Google Drive cloud storage",
            "MongoDB record keeping",
            "Automatic local file cleanup"
        ],
        "endpoints": {
            "process": "/process-video/",
            "upload": "/upload-and-process/",
            "status": "/status/{task_id}",
            "tasks": "/tasks/",
            "user_videos": "/users/{user_id}/videos/",
            "video_details": "/videos/{session_id}",
            "stats": "/stats/"
        }
    }

@app.post("/process-video/")
async def process_video_endpoint(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """
    Process a video file that exists on the server filesystem
    """
    if not os.path.exists(request.input_video_path):
        raise HTTPException(status_code=404, detail="Input video file not found")
    
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "progress": "Task queued for processing",
        "user_id": request.user_id
    }
    
    # Start background processing
    background_tasks.add_task(
        process_video_background, 
        task_id,
        request.input_video_path,
        request.user_id,
        request.output_video_path,
        request.keep_intermediate_files,
        request.config
    )
    
    logger.info(f"Started processing task {task_id} for user {request.user_id}, video: {request.input_video_path}")
    return {
        "task_id": task_id, 
        "status": "started",
        "user_id": request.user_id,
        "message": "Video will be uploaded to Google Drive and stored in database upon completion"
    }

@app.post("/upload-and-process/")
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    video_file: UploadFile = File(...),
    keep_intermediate_files: bool = Form(False),
    config: str = Form("{}")  # JSON string for config
):
    """
    Upload a video file and process it with cloud storage
    """
    import json
    
    # The user_id is now guaranteed by Form(...) so this check is redundant but safe
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
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "uploading",
        "result": None,
        "error": None,
        "progress": "Uploading video file",
        "user_id": user_id
    }
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_input_path = temp_file.name
        shutil.copyfileobj(video_file.file, temp_file)
    
    logger.info(f"Uploaded video saved to: {temp_input_path}")
    
    # Start background processing
    background_tasks.add_task(
        process_uploaded_video_background,
        task_id,
        temp_input_path,
        user_id,
        keep_intermediate_files,
        config_dict
    )
    
    return {
        "task_id": task_id, 
        "status": "started", 
        "filename": video_file.filename,
        "user_id": user_id,
        "message": "Video will be uploaded to Google Drive and stored in database upon completion"
    }

async def process_video_background(
    task_id: str, 
    input_video_path: str,
    user_id: str,
    output_video_path: Optional[str],
    keep_intermediate_files: bool,
    config: Dict[str, Any]
):
    """Background task for processing video files with cloud storage"""
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = "Starting video transcription and optimization"
        
        result = await process_video_request(
            input_video_path=input_video_path,
            user_id=user_id,
            output_video_path=output_video_path,
            keep_intermediate_files=keep_intermediate_files,
            config=config
        )
        
        if result["success"]:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["result"] = result
            tasks[task_id]["progress"] = "Processing completed - video uploaded to Google Drive"
            logger.info(f"Task {task_id} completed successfully for user {user_id}")
            logger.info(f"Google Drive File ID: {result['gdrive']['file_id']}")
        else:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = result.get("error", "Unknown error")
            tasks[task_id]["progress"] = "Processing failed"
            logger.error(f"Task {task_id} failed for user {user_id}: {result.get('error')}")
            
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["progress"] = "Processing failed with exception"
        logger.error(f"Task {task_id} failed with exception for user {user_id}: {e}")

async def process_uploaded_video_background(
    task_id: str,
    temp_input_path: str,
    user_id: str,
    keep_intermediate_files: bool,
    config: Dict[str, Any]
):
    """Background task for processing uploaded video files"""
    try:
        await process_video_background(
            task_id, 
            temp_input_path,
            user_id,
            None,  # output_video_path 
            keep_intermediate_files, 
            config
        )
    finally:
        # Clean up uploaded temporary file
        try:
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
                logger.info(f"Cleaned up uploaded file: {temp_input_path}")
        except Exception as e:
            logger.warning(f"Could not clean up uploaded file {temp_input_path}: {e}")

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get the status of a processing task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return ProcessingStatus(
        task_id=task_id,
        status=tasks[task_id]["status"],
        result=tasks[task_id]["result"],
        error=tasks[task_id]["error"],
        progress=tasks[task_id]["progress"]
    )

@app.get("/tasks/")
async def list_tasks():
    """List all tasks and their statuses"""
    return {
        "total_tasks": len(tasks),
        "tasks": {
            task_id: {
                "status": task_data["status"],
                "progress": task_data["progress"],
                "user_id": task_data.get("user_id")
            }
            for task_id, task_data in tasks.items()
        }
    }

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a completed or failed task from memory"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_status = tasks[task_id]["status"]
    if task_status in ["processing", "pending", "uploading"]:
        raise HTTPException(status_code=400, detail="Cannot delete active task")
    
    del tasks[task_id]
    return {"message": f"Task {task_id} deleted successfully"}

@app.get("/users/{user_id}/videos/")
async def get_user_videos_endpoint(
    user_id: str, 
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get all videos processed for a specific user"""
    try:
        videos = await get_user_videos(user_id, limit, offset)
        return {
            "user_id": user_id,
            "total_returned": len(videos),
            "limit": limit,
            "offset": offset,
            "videos": videos
        }
    except Exception as e:
        logger.error(f"Error retrieving videos for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/videos/{session_id}")
async def get_video_details(session_id: str):
    """Get detailed information about a specific video by session ID"""
    try:
        video = await get_video_by_session(session_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return video
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving video {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats/")
async def get_stats(user_id: Optional[str] = None):
    """Get processing statistics, optionally filtered by user"""
    try:
        stats = await get_processing_stats(user_id)
        return {
            "user_id": user_id if user_id else "all_users",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/process-multiple/")
async def process_multiple_videos(
    requests: list[ProcessingRequest],
    background_tasks: BackgroundTasks,
    max_concurrent: int = 2
):
    """
    Process multiple videos concurrently with cloud storage
    """
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 videos per batch")
    
    # Validate all requests first
    for request in requests:
        if not os.path.exists(request.input_video_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Input video file not found: {request.input_video_path}"
            )
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id is required for all requests")
    
    task_ids = []
    
    for request in requests:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "progress": "Queued for batch processing",
            "user_id": request.user_id
        }
    
    # Start batch processing
    background_tasks.add_task(
        process_batch_background,
        task_ids,
        requests,
        max_concurrent
    )
    
    return {
        "message": f"Started batch processing of {len(requests)} videos",
        "task_ids": task_ids,
        "max_concurrent": max_concurrent,
        "note": "All videos will be uploaded to Google Drive and stored in database"
    }

async def process_batch_background(
    task_ids: list[str],
    requests: list[ProcessingRequest],
    max_concurrent: int
):
    """Background task for batch processing with cloud storage"""
    from main_video_processor import process_multiple_videos_concurrently
    
    # Convert requests to the format expected by process_multiple_videos_concurrently
    video_requests = []
    for i, request in enumerate(requests):
        video_requests.append({
            'input_path': request.input_video_path,
            'user_id': request.user_id,
            'output_path': request.output_video_path,
            'keep_intermediate_files': request.keep_intermediate_files,
            'config': request.config
        })
        tasks[task_ids[i]]["status"] = "processing"
        tasks[task_ids[i]]["progress"] = "Processing in batch"
    
    try:
        results = await process_multiple_videos_concurrently(video_requests, max_concurrent)
        
        # Update task statuses with results
        for i, (task_id, result) in enumerate(zip(task_ids, results)):
            if isinstance(result, Exception):
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = str(result)
                tasks[task_id]["progress"] = "Batch processing failed"
            elif result.get('success'):
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["result"] = result
                tasks[task_id]["progress"] = "Batch processing completed - uploaded to Google Drive"
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = result.get("error", "Unknown error")
                tasks[task_id]["progress"] = "Batch processing failed"
                
    except Exception as e:
        # Mark all tasks as failed
        for task_id in task_ids:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = f"Batch processing error: {str(e)}"
            tasks[task_id]["progress"] = "Batch processing failed"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )

# Run the server with:
# uvicorn updated_server:app --host 0.0.0.0 --port 8000 --reload