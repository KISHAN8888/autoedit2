import os
import tempfile
import shutil
import asyncio
import logging
from celery import current_task
from datetime import datetime
from app.workers.celery_app import celery_app
from app.database.mongodb_manager import MongoDBManager
from app.services.google_drive_uploader import GoogleDriveUploader
from app.config import get_settings

# Import video processing modules
from app.services.video_summarizer2 import VideoSummarizer, SRTSegment, OptimizedSegment

logger = logging.getLogger(__name__)
settings = get_settings()

@celery_app.task(bind=True)
def process_video_with_ffmpeg(self, task_id: str, input_video_path: str, segment_audio_data: list, config: dict):
    """
    FFmpeg worker - handles CPU-intensive video operations
    LOW CONCURRENCY (max 2) - queued processing for resource management
    """
    logger.info(f"FFmpeg worker started for task {task_id}")
    
    try:
        current_task.update_state(
            state='PROGRESS', 
            meta={'progress': 0, 'status': 'Starting FFmpeg processing'}
        )
        
        # Initialize async components
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run async processing
        result = loop.run_until_complete(
            _process_ffmpeg_async(task_id, input_video_path, segment_audio_data, config, self)
        )
        
        loop.close()
        return result
        
    except Exception as e:
        logger.error(f"FFmpeg worker failed for task {task_id}: {e}")
        
        # Update MongoDB on failure
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
            loop.run_until_complete(mongodb_manager.initialize())
            loop.run_until_complete(mongodb_manager.update_task_status(task_id, "failed", error=str(e)))
            loop.run_until_complete(mongodb_manager.close())
        except Exception as db_error:
            logger.error(f"Failed to update MongoDB after FFmpeg failure: {db_error}")
        finally:
            loop.close()
        
        raise

async def _process_ffmpeg_async(task_id: str, input_video_path: str, segment_audio_data: list, config: dict, celery_task):
    """Async FFmpeg processing - video operations only"""
    
    mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
    await mongodb_manager.initialize()
    
    try:
        logger.info(f"FFmpeg processing {len(segment_audio_data)} segments for task {task_id}")
        
        # Update status
        await mongodb_manager.update_task_status(task_id, "video_processing")
        celery_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Processing video segments'})
        
        # Initialize summarizer for FFmpeg operations only
        summarizer = VideoSummarizer(
            azure_openai_key=settings.azure_openai_api_key,
            azure_openai_endpoint=settings.azure_openai_endpoint,
            azure_api_version=settings.azure_openai_api_version,
            azure_deployment_name=settings.azure_openai_deployment_name,
            murf_api_key=settings.murf_api_key,
            tts_engine=config.get('tts_engine', settings.tts_engine),
            target_wpm=config.get('target_wpm', settings.target_wpm)
        )
        
        # Create temporary directory for FFmpeg processing
        with tempfile.TemporaryDirectory() as worker_temp_dir:
            logger.info(f"Using temporary directory: {worker_temp_dir}")
            
            # Process video segments with FFmpeg
            celery_task.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Extracting and retiming segments'})
            
            semaphore = asyncio.Semaphore(2)  # Limit concurrent FFmpeg processes
            
            async def process_single_segment(segment_data: dict):
                async with semaphore:
                    i = segment_data["index"]
                    opt_seg_data = segment_data["optimized_segment"]
                    actual_audio_duration = segment_data["audio_duration"]
                    audio_path = segment_data["audio_path"]
                    
                    # Reconstruct optimized segment
                    original = SRTSegment(**opt_seg_data["original"])
                    opt_seg = OptimizedSegment(
                        original=original,
                        optimized_text=opt_seg_data["optimized_text"],
                        optimized_word_count=opt_seg_data["optimized_word_count"],
                        estimated_speech_duration=opt_seg_data["estimated_speech_duration"],
                        speed_multiplier=opt_seg_data["speed_multiplier"],
                        reasoning=opt_seg_data["reasoning"]
                    )

                    segment_temp_dir = os.path.join(worker_temp_dir, f"segment_{i}")
                    os.makedirs(segment_temp_dir, exist_ok=True)
                    segment_video_path = os.path.join(segment_temp_dir, f"video_{i:04d}.mp4")
                    
                    start_time = opt_seg.original.start_seconds
                    original_duration = opt_seg.original.duration
                    speed_multiplier = original_duration / actual_audio_duration

                    try:
                        # FFmpeg: Extract and retime video segment
                        success = await summarizer.ffmpeg_processor.extract_and_retime_segment(
                            input_video_path, segment_video_path, start_time,
                            original_duration, speed_multiplier
                        )
                        if success:
                            logger.debug(f"Segment {i+1}: Speed {speed_multiplier:.2f}x")
                            return segment_video_path, audio_path
                        else:
                            return None
                    except Exception as e:
                        logger.error(f"Error processing segment {i}: {e}")
                        return None
            
            # Process all segments
            video_tasks = [process_single_segment(data) for data in segment_audio_data]
            results = await asyncio.gather(*video_tasks, return_exceptions=True)
            
            # Collect successful results
            video_segments = []
            audio_segments = []
            for result in results:
                if result and not isinstance(result, Exception):
                    video_path, audio_path = result
                    video_segments.append(video_path)
                    audio_segments.append(audio_path)
            
            if not video_segments:
                raise ValueError("No valid video segments were created")

            logger.info(f"Successfully processed {len(video_segments)} video segments")
            
            # Combine segments
            celery_task.update_state(state='PROGRESS', meta={'progress': 60, 'status': 'Combining segments'})
            
            combined_video = os.path.join(worker_temp_dir, "combined.mp4")
            await summarizer._robust_combine_segments_async(video_segments, audio_segments, combined_video, worker_temp_dir)
            
            # Move to persistent location
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dest_f:
                final_video_path = dest_f.name
            shutil.move(combined_video, final_video_path)
            
            # Apply background if needed
            celery_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Applying effects'})
            
            background_image_path = config.get('background_image_path')
            if background_image_path and os.path.exists(background_image_path):
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    background_video_path = temp_file.name
                
                success = await summarizer._apply_background_async(
                    final_video_path, background_image_path, background_video_path, 
                    config.get('overlay_options', {})
                )
                
                if success:
                    os.remove(final_video_path)
                    final_video_path = background_video_path
        
        # Upload to Google Drive
        celery_task.update_state(state='PROGRESS', meta={'progress': 90, 'status': 'Uploading to cloud'})
        
        gdrive_uploader = GoogleDriveUploader(
            service_account_path=settings.google_service_account_path,
            folder_id=config.get('gdrive_folder_id', settings.gdrive_folder_id)
        )
        
        upload_filename = f"processed_{task_id}.mp4"
        gdrive_result = await gdrive_uploader.upload_video_async(final_video_path, upload_filename)
        
        if not gdrive_result['success']:
            raise Exception(f"Google Drive upload failed: {gdrive_result['error']}")
        
        logger.info(f"Video uploaded to Google Drive: {gdrive_result['file_id']}")
        
        # Store final results
        cost_summary = summarizer.get_cost_summary()
        
        result_data = {
            "gdrive_data": gdrive_result,
            "cost_summary": cost_summary,
            "completed_at": datetime.utcnow(),
            "processing_details": {
                "segments_processed": len(video_segments),
                "total_segments": len(segment_audio_data)
            }
        }
        
        await mongodb_manager.update_task_status(task_id, "completed", result=result_data)
        
        # Cleanup
        try:
            os.remove(final_video_path)
            upload_dir = os.path.dirname(input_video_path)
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
        
        celery_task.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Completed'})
        
        logger.info(f"Task {task_id} completed successfully")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "gdrive_file_id": gdrive_result["file_id"],
            "gdrive_view_link": gdrive_result["web_view_link"],
            "segments_processed": len(video_segments)
        }
        
    finally:
        await mongodb_manager.close()