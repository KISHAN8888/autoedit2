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
from app.services.cloudflarer2_uploader import CloudflareR2Uploader
from app.config import get_settings

# Import video processing modules
from app.services.video_summarizer2 import VideoSummarizer, SRTSegment, SentenceGroup

logger = logging.getLogger(__name__)
settings = get_settings()

@celery_app.task(bind=True)
def process_video_with_ffmpeg(self, task_id: str, input_video_path: str, sentence_groups_data: list, config: dict):
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
            _process_ffmpeg_async(task_id, input_video_path, sentence_groups_data, config, self)
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

async def _process_ffmpeg_async(task_id: str, input_video_path: str, sentence_groups_data: list, config: dict, celery_task):
    """Async FFmpeg processing - video operations only"""
    
    mongodb_manager = MongoDBManager(settings.mongodb_uri, settings.mongodb_database)
    await mongodb_manager.initialize()
    uploader = CloudflareR2Uploader()
    try:
        logger.info(f"FFmpeg processing {len(sentence_groups_data)} sentence groups for task {task_id}")
        
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
            target_wpm=config.get('target_wpm', settings.target_wpm),
            groq_playai_config=settings.groq_playai_config, 
            vibe_voice_config=settings.vibe_voice_config,            
        )
        
        # Create temporary directory for FFmpeg processing
        with tempfile.TemporaryDirectory() as worker_temp_dir:
            logger.info(f"Using temporary directory: {worker_temp_dir}")
            
            # Process video segments with FFmpeg - MANUAL PROCESSING (no TTS regeneration)
            celery_task.update_state(state='PROGRESS', meta={'progress': 30, 'status': 'Processing sentence groups'})
            
            semaphore = asyncio.Semaphore(4)
            final_clips_for_concat = {}  # Use dict to handle any index values
            
            async def process_single_group_ffmpeg_only(group_data: dict):
                async with semaphore:
                    i = group_data["index"]
                    segment_temp_dir = os.path.join(worker_temp_dir, f"group_{i}")
                    os.makedirs(segment_temp_dir, exist_ok=True)
                    
                    # Use EXISTING audio file (no regeneration)
                    existing_audio_path = group_data["audio_path"]
                    existing_audio_duration = group_data["audio_duration"]
                    
                    if not os.path.exists(existing_audio_path):
                        logger.warning(f"Skipping group {i} - audio file not found: {existing_audio_path}")
                        return
    
                    speed_multiplier = group_data["original_duration"] / existing_audio_duration
                    
                    retimed_video_path = os.path.join(segment_temp_dir, f"retimed_video_{i}.mp4")
                    extract_success = await summarizer.ffmpeg_processor.extract_and_retime_segment(
                        input_video_path, retimed_video_path, group_data["start_seconds"],
                        group_data["original_duration"], speed_multiplier
                    )
                    
                    if not extract_success:
                        logger.warning(f"Skipping group {i} due to video re-timing failure.")
                        return
    
                    final_clip_path = os.path.join(worker_temp_dir, f"final_clip_{i:04d}.mp4")
                    combine_success = await summarizer.ffmpeg_processor.combine_video_audio_segment(
                        retimed_video_path, existing_audio_path, final_clip_path, existing_audio_duration
                    )
                    
                    if combine_success:
                        logger.info(f"Group {i+1}/{len(sentence_groups_data)} processed. Speed: {speed_multiplier:.2f}x")
                        final_clips_for_concat[i] = final_clip_path
                    else:
                        logger.warning(f"Skipping group {i} due to final combination failure.")
            
            # Process all groups using existing audio
            tasks = [process_single_group_ffmpeg_only(group_data) for group_data in sentence_groups_data]
            await asyncio.gather(*tasks)
    
            # Sort by index to maintain proper order for concatenation
            valid_final_clips = []
            for index in sorted(final_clips_for_concat.keys()):
                if final_clips_for_concat[index]:
                    valid_final_clips.append(final_clips_for_concat[index])
            if not valid_final_clips:
                raise ValueError("No video clips were successfully processed.")
    
            concat_list_path = os.path.join(worker_temp_dir, "concat_list.txt")
            with open(concat_list_path, 'w') as f:
                for clip_path in valid_final_clips:
                    f.write(f"file '{os.path.abspath(clip_path)}'\n")
    
            internal_output_video = os.path.join(worker_temp_dir, "final_video.mp4")
            concat_success = await summarizer.ffmpeg_processor.concatenate_files(concat_list_path, internal_output_video)
    
            if not concat_success:
                raise RuntimeError("Final video concatenation failed.")
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dest_f:
                final_video_path = dest_f.name
            shutil.move(internal_output_video, final_video_path)
            
            logger.info(f"Successfully processed all sentence groups")
            
            # Apply background if needed
            celery_task.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Applying effects'})
            
            # background_image_path = config.get('background_image_path')
            # if background_image_path and os.path.exists(background_image_path):
            #     with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            #         background_video_path = temp_file.name
                
            #     # Apply background if the method exists (you may need to implement this)
            #     # For now, we'll skip background application since it's not in your updated code
            #     logger.info("Background application skipped - method not available in updated summarizer")
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
        #gdrive_result = await gdrive_uploader.upload_video_async(final_video_path, upload_filename)
        upload_result = await uploader.upload_video_async(final_video_path, upload_filename)
        #if not gdrive_result['success']:
        #    raise Exception(f"Google Drive upload failed: {gdrive_result['error']}")
        #
        #logger.info(f"Video uploaded to Google Drive: {gdrive_result['file_id']}")
        
        # Store final results
        cost_summary = summarizer.get_cost_summary()
        
        result_data = {
            "gdrive_data": upload_result,
            "cost_summary": cost_summary,
            "completed_at": datetime.utcnow(),
            "processing_details": {
                "sentence_groups_processed": len(valid_final_clips),
                "total_sentence_groups": len(sentence_groups_data)
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
            "gdrive_file_id": upload_result["file_id"],
            "gdrive_view_link": upload_result["web_view_link"],
            "sentence_groups_processed": len(valid_final_clips)
        }
        
    finally:
        await mongodb_manager.close()