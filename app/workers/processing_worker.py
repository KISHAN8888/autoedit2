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