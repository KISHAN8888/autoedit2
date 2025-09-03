# # app/workers/video_worker.py
# import os
# import tempfile
# import shutil
# import asyncio
# import logging
# from celery import current_task
# from datetime import datetime
# from app.workers.celery_app import celery_app
# from app.database.mongodb_manager import MongoDBManager
# from app.services.google_drive_uploader import GoogleDriveUploader
# from app.config import get_settings

# # Import your video processing modules - update paths as needed
# from app.services.video_summarizer2 import VideoSummarizer, SRTSegment, OptimizedSegment

# logger = logging.getLogger(__name__)
# settings = get_settings()

# # @celery_app.task(bind=True)
# # def process_video_with_ffmpeg(self, task_id: str, input_video_path: str, segment_audio_data: list, config: dict):
# #     """
# #     Celery worker task for FFmpeg video processing ONLY
# #     Receives pre-generated audio files and processes video segments
# #     """
# #     logger.info(f"Starting FFmpeg processing for task {task_id}")
    
# #     try:
# #         # Update task status
# #         #current_task.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting FFmpeg processing'})
# #         self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting FFmpeg processing'})
# #         # Initialize async components
# #         loop = asyncio.new_event_loop()
# #         asyncio.set_event_loop(loop)
        
# #         # Run async processing
# #         result = loop.run_until_complete(
# #             _process_video_ffmpeg_async(task_id, input_video_path, segment_audio_data, config, self)
# #         )
        
# #         loop.close()
# #         return result
        
# #     except Exception as e:
# #         logger.error(f"FFmpeg processing failed for task {task_id}: {e}")
        
# #         # Update task status in MongoDB on failure
# #         loop = asyncio.new_event_loop()
# #         asyncio.set_event_loop(loop)
        
# #         try:
# #             mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
# #             loop.run_until_complete(mongodb_manager.initialize())
# #             loop.run_until_complete(mongodb_manager.update_task_status(task_id, "failed", error=str(e)))
# #             loop.run_until_complete(mongodb_manager.close())
# #         except Exception as db_error:
# #             logger.error(f"Failed to update MongoDB after task failure: {db_error}")
# #         finally:
# #             loop.close()
        
# #         raise

# @celery_app.task
# def process_video_with_ffmpeg(task_id: str, input_video_path: str, segment_audio_data: list, config: dict):
#     """
#     Celery worker task for FFmpeg video processing ONLY
#     Receives pre-generated audio files and processes video segments
#     """
#     logger.info(f"Starting FFmpeg processing for task {task_id}")
    
#     try:
#         # Update task status
#         from celery import current_task
#         current_task.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting FFmpeg processing'})
        
#         # Initialize async components
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
        
#         # Run async processing
#         result = loop.run_until_complete(
#             _process_video_ffmpeg_async(task_id, input_video_path, segment_audio_data, config, current_task)
#         )
        
#         loop.close()
#         return result
        
#     except Exception as e:
#         logger.error(f"FFmpeg processing failed for task {task_id}: {e}")
        
#         # Update task status in MongoDB on failure
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
        
#         try:
#             mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
#             loop.run_until_complete(mongodb_manager.initialize())
#             loop.run_until_complete(mongodb_manager.update_task_status(task_id, "failed", error=str(e)))
#             loop.run_until_complete(mongodb_manager.close())
#         except Exception as db_error:
#             logger.error(f"Failed to update MongoDB after task failure: {db_error}")
#         finally:
#             loop.close()
        
#         raise

# async def _process_video_ffmpeg_async(task_id: str, input_video_path: str, segment_audio_data: list, config: dict, celery_task):
#     """Async FFmpeg video processing function - ONLY FFmpeg operations"""
    
#     mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
#     await mongodb_manager.initialize()
    
#     try:
#         logger.info(f"Processing {len(segment_audio_data)} segments for task {task_id}")
        
#         # Update status
#         await mongodb_manager.update_task_status(task_id, "video_processing")
#         celery_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Processing video segments with FFmpeg'})
        
#         # Initialize FFmpeg processor (no TTS, only FFmpeg operations)
#         summarizer = VideoSummarizer(
#             azure_openai_key=settings.azure_openai_api_key,
#             azure_openai_endpoint=settings.azure_openai_endpoint,
#             azure_api_version=settings.azure_openai_api_version,
#             azure_deployment_name=settings.azure_openai_deployment_name,
#             murf_api_key=settings.murf_api_key,
#             tts_engine=config.get('tts_engine', settings.tts_engine),
#             target_wpm=config.get('target_wpm', settings.target_wpm)
#         )
        
#         # Create temporary directory for this worker session
#         with tempfile.TemporaryDirectory() as worker_temp_dir:
#             logger.info(f"Using temporary directory: {worker_temp_dir}")
            
#             # --- STAGE 2: PROCESS ALL VIDEO SEGMENTS CONCURRENTLY (FFmpeg only) ---
#             celery_task.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Extracting and retiming video segments'})
            
#             semaphore = asyncio.Semaphore(2)  # Limit concurrent FFmpeg processes

#             async def process_single_video_segment(segment_data: dict):
#                 async with semaphore:
#                     i = segment_data["index"]
#                     opt_seg_data = segment_data["optimized_segment"]
#                     actual_audio_duration = segment_data["audio_duration"]
#                     audio_path = segment_data["audio_path"]
                    
#                     # Reconstruct optimized segment
#                     original = SRTSegment(**opt_seg_data["original"])
#                     opt_seg = OptimizedSegment(
#                         original=original,
#                         optimized_text=opt_seg_data["optimized_text"],
#                         optimized_word_count=opt_seg_data["optimized_word_count"],
#                         estimated_speech_duration=opt_seg_data["estimated_speech_duration"],
#                         speed_multiplier=opt_seg_data["speed_multiplier"],
#                         reasoning=opt_seg_data["reasoning"]
#                     )

#                     segment_temp_dir = os.path.join(worker_temp_dir, f"segment_{i}")
#                     os.makedirs(segment_temp_dir, exist_ok=True)
#                     segment_video_path = os.path.join(segment_temp_dir, f"video_{i:04d}.mp4")
                    
#                     start_time = opt_seg.original.start_seconds
#                     original_video_seg_duration = opt_seg.original.duration
#                     speed_multiplier = original_video_seg_duration / actual_audio_duration

#                     try:
#                         # FFmpeg: Extract and retime video segment
#                         success = await summarizer.ffmpeg_processor.extract_and_retime_segment(
#                             input_video_path, segment_video_path, start_time,
#                             original_video_seg_duration, speed_multiplier
#                         )
#                         if success:
#                             logger.info(f"Segment {i+1}: Speed {speed_multiplier:.2f}x, Duration: {actual_audio_duration:.2f}s")
#                             return segment_video_path, audio_path
#                         else:
#                             logger.warning(f"Failed to process video segment {i}")
#                             return None
#                     except Exception as e:
#                         logger.error(f"Exception processing video segment {i}: {e}")
#                         return None

#             # Process all video segments
#             video_tasks = [process_single_video_segment(data) for data in segment_audio_data]
#             video_processing_results = await asyncio.gather(*video_tasks, return_exceptions=True)
            
#             celery_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Combining video segments'})
            
#             # --- STAGE 3: COLLECT RESULTS AND COMBINE (FFmpeg only) ---
#             video_segments = []
#             audio_segments = []
#             for result in video_processing_results:
#                 if result and not isinstance(result, Exception):
#                     video_path, audio_path = result
#                     video_segments.append(video_path)
#                     audio_segments.append(audio_path)
            
#             if not video_segments:
#                 raise ValueError("No valid video segments were created after FFmpeg processing.")

#             logger.info(f"Successfully processed {len(video_segments)} video segments")

#             # FFmpeg: Combine all processed segments
#             internal_output_video = os.path.join(worker_temp_dir, "final_output.mp4")
#             await summarizer._robust_combine_segments_async(video_segments, audio_segments, internal_output_video, worker_temp_dir)
            
#             # Move to persistent location
#             with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dest_f:
#                 persistent_output_path = dest_f.name
#             shutil.move(internal_output_video, persistent_output_path)
            
#             logger.info(f"Video combination complete: {persistent_output_path}")
            
#             celery_task.update_state(state='PROGRESS', meta={'progress': 70, 'status': 'Applying background'})
            
#             # FFmpeg: Apply background if specified
#             final_video_path = persistent_output_path
#             background_image_path = config.get('background_image_path')
#             overlay_options = config.get('overlay_options', {})
            
#             if background_image_path and os.path.exists(background_image_path):
#                 logger.info(f"Applying background: {background_image_path}")
#                 with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
#                     background_video_path = temp_file.name
                
#                 success = await summarizer._apply_background_async(
#                     persistent_output_path,
#                     background_image_path,
#                     background_video_path,
#                     overlay_options
#                 )
                
#                 if success:
#                     os.remove(persistent_output_path)
#                     final_video_path = background_video_path
#                     logger.info("Background applied successfully")
#                 else:
#                     logger.warning("Background application failed, using original video")
            
#         celery_task.update_state(state='PROGRESS', meta={'progress': 85, 'status': 'Uploading to Google Drive'})
        
#         # Upload to Google Drive
#         gdrive_uploader = GoogleDriveUploader(
#             service_account_path=settings.google_service_account_path,
#             folder_id=config.get('gdrive_folder_id', settings.gdrive_folder_id)
#         )
        
#         upload_filename = f"processed_{task_id}.mp4"
#         gdrive_result = await gdrive_uploader.upload_video_async(final_video_path, upload_filename)
        
#         if not gdrive_result['success']:
#             raise Exception(f"Google Drive upload failed: {gdrive_result['error']}")
        
#         logger.info(f"Video uploaded to Google Drive: {gdrive_result['file_id']}")
        
#         celery_task.update_state(state='PROGRESS', meta={'progress': 95, 'status': 'Finalizing'})
        
#         # Get cost summary
#         cost_summary = summarizer.get_cost_summary()
        
#         # Store final results in MongoDB
#         result_data = {
#             "gdrive_data": gdrive_result,
#             "cost_summary": cost_summary,
#             "completed_at": datetime.utcnow(),
#             "processing_details": {
#                 "segments_processed": len(video_segments),
#                 "total_segments": len(segment_audio_data),
#                 "background_applied": bool(background_image_path and os.path.exists(background_image_path or ""))
#             }
#         }
        
#         await mongodb_manager.update_task_status(task_id, "completed", result=result_data)
        
#         # Cleanup local files
#         try:
#             os.remove(final_video_path)
#             logger.info("Local video file cleaned up")
            
#             # Clean up original upload directory if it exists
#             if os.path.exists(input_video_path):
#                 upload_dir = os.path.dirname(input_video_path)
#                 if os.path.exists(upload_dir):
#                     shutil.rmtree(upload_dir, ignore_errors=True)
#                     logger.info("Upload directory cleaned up")
                    
#         except Exception as cleanup_error:
#             logger.warning(f"Cleanup error: {cleanup_error}")
        
#         celery_task.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Completed'})
        
#         logger.info(f"Task {task_id} completed successfully")
        
#         return {
#             "task_id": task_id,
#             "status": "completed",
#             "gdrive_file_id": gdrive_result["file_id"],
#             "gdrive_view_link": gdrive_result["web_view_link"],
#             "cost": cost_summary["estimated_cost_usd"],
#             "segments_processed": len(video_segments)
#         }
        
#     finally:
#         await mongodb_manager.close()

# @celery_app.task(bind=True)
# def cleanup_old_tasks(self):
#     """Periodic task to cleanup old completed/failed tasks"""
#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
        
#         mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
#         loop.run_until_complete(mongodb_manager.initialize())
        
#         # Cleanup tasks older than 30 days
#         deleted_count = loop.run_until_complete(mongodb_manager.cleanup_old_tasks(30))
        
#         loop.run_until_complete(mongodb_manager.close())
#         loop.close()
        
#         logger.info(f"Cleanup completed: {deleted_count} tasks removed")
#         return {"deleted_count": deleted_count}
        
#     except Exception as e:
#         logger.error(f"Cleanup task failed: {e}")
#         raise


import os
import tempfile
import shutil
import asyncio
import logging
from celery import current_task
from datetime import datetime
from app.workers.celery_app import celery_app
from app.database.mongodb_manager import MongoDBManager
from app.services.video_processor import AsyncVideoProcessor
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@celery_app.task(bind=True)
def process_video_content(self, task_id: str, input_video_path: str, user_id: str, config: dict):
    """
    Processing worker - handles transcription, AI optimization, and audio generation
    HIGH CONCURRENCY - processes multiple requests simultaneously
    """
    logger.info(f"Processing worker started for task {task_id}")
    
    try:
        current_task.update_state(
            state='PROGRESS', 
            meta={'progress': 0, 'status': 'Starting content processing'}
        )
        
        # Initialize async components
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run async processing
        result = loop.run_until_complete(
            _process_content_async(task_id, input_video_path, user_id, config, self)
        )
        
        loop.close()
        return result
        
    except Exception as e:
        logger.error(f"Processing worker failed for task {task_id}: {e}")
        
        # Update MongoDB on failure
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
            loop.run_until_complete(mongodb_manager.initialize())
            loop.run_until_complete(mongodb_manager.update_task_status(task_id, "failed", error=str(e)))
            loop.run_until_complete(mongodb_manager.close())
        except Exception as db_error:
            logger.error(f"Failed to update MongoDB after processing failure: {db_error}")
        finally:
            loop.close()
        
        raise

async def _process_content_async(task_id: str, input_video_path: str, user_id: str, config: dict, celery_task):
    """Async content processing - transcription, optimization, audio generation"""
    
    mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
    await mongodb_manager.initialize()
    
    try:
        logger.info(f"Starting content processing for task {task_id}")
        
        # Initialize processor
        processor = AsyncVideoProcessor(config)
        
        # Step 1: Transcription
        await mongodb_manager.update_task_status(task_id, "transcribing")
        celery_task.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Transcribing video'})
        
        language, srt_content = await processor.transcriber.transcribe_video_async(input_video_path)
        
        if not srt_content or not srt_content.strip():
            raise ValueError("Transcription failed - no content generated")
        
        # Step 2: AI Optimization
        await mongodb_manager.update_task_status(task_id, "optimizing")
        celery_task.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Optimizing script'})
        
        segments = processor.summarizer.parse_srt_content(srt_content)
        optimized_segments = await processor.summarizer.optimize_all_segments_async(segments)
        
        # Step 3: Audio Generation (HIGH CONCURRENCY - can handle many simultaneously)
        await mongodb_manager.update_task_status(task_id, "generating_audio")
        celery_task.update_state(state='PROGRESS', meta={'progress': 50, 'status': 'Generating audio'})
        
        # Get upload directory from input path
        upload_dir = os.path.dirname(input_video_path)
        segment_audio_data = []
        audio_tasks = []
        
        # Generate audio for all segments concurrently
        for i, opt_seg in enumerate(optimized_segments):
            segment_dir = os.path.join(upload_dir, f"segment_{i}")
            os.makedirs(segment_dir, exist_ok=True)
            audio_tasks.append(
                processor.summarizer.generate_tts_audio_async(opt_seg.optimized_text, segment_dir)
            )
        
        audio_results = await asyncio.gather(*audio_tasks, return_exceptions=True)
        
        # Process audio results
        for i, result in enumerate(audio_results):
            if isinstance(result, Exception) or not result or not result[0]:
                logger.warning(f"Audio generation failed for segment {i}")
                continue
            
            audio_path, audio_duration = result
            segment_audio_data.append({
                "index": i,
                "audio_path": audio_path,
                "audio_duration": audio_duration,
                "optimized_segment": serialize_optimized_segment(optimized_segments[i])
            })
        
        if not segment_audio_data:
            raise ValueError("All audio generation failed")
        
        logger.info(f"Generated audio for {len(segment_audio_data)}/{len(optimized_segments)} segments")
        
        celery_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Queueing for video processing'})
        
        # Store processing results
        processing_data = {
            "transcription": {
                "language": language,
                "content_length": len(srt_content)
            },
            "optimization": {
                "segments_processed": len(optimized_segments),
                "segments_with_audio": len(segment_audio_data)
            }
        }
        await mongodb_manager.update_task_data(task_id, processing_data)
        
        # Queue for FFmpeg worker (LOW CONCURRENCY, QUEUED)
        await mongodb_manager.update_task_status(task_id, "queued_for_video_processing")
        
        ffmpeg_task = celery_app.send_task(
            'app.workers.ffmpeg_worker.process_video_with_ffmpeg',
            args=[task_id, input_video_path, segment_audio_data, config],
            task_id=f"ffmpeg_{task_id}",
            queue='ffmpeg'  # Separate queue for FFmpeg worker
        )
        
        await mongodb_manager.update_task_data(task_id, {"ffmpeg_task_id": ffmpeg_task.id})
        
        celery_task.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Queued for video processing'})
        
        logger.info(f"Processing complete for task {task_id}, queued for FFmpeg: {ffmpeg_task.id}")
        
        return {
            "task_id": task_id,
            "status": "queued_for_video_processing",
            "segments_processed": len(optimized_segments),
            "segments_with_audio": len(segment_audio_data),
            "ffmpeg_task_id": ffmpeg_task.id
        }
        
    finally:
        await mongodb_manager.close()

def serialize_optimized_segment(opt_seg):
    """Serialize OptimizedSegment for FFmpeg worker"""
    return {
        "original": {
            "index": opt_seg.original.index,
            "start_time": opt_seg.original.start_time,
            "end_time": opt_seg.original.end_time,
            "start_seconds": opt_seg.original.start_seconds,
            "end_seconds": opt_seg.original.end_seconds,
            "duration": opt_seg.original.duration,
            "text": opt_seg.original.text,
            "word_count": opt_seg.original.word_count,
            "words_per_second": opt_seg.original.words_per_second
        },
        "optimized_text": opt_seg.optimized_text,
        "optimized_word_count": opt_seg.optimized_word_count,
        "estimated_speech_duration": opt_seg.estimated_speech_duration,
        "speed_multiplier": opt_seg.speed_multiplier,
        "reasoning": opt_seg.reasoning
    }