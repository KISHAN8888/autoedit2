# import os
# import sys
# import logging
# import tempfile
# import shutil
# import asyncio
# from pathlib import Path
# from dotenv import load_dotenv
# from video_transcriber import VideoTranscriber
# from video_summarizer import VideoSummarizer

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('video_processing.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# class AsyncVideoProcessor:
#     """Async main orchestrator for video transcription and summarization pipeline"""
    
#     def __init__(self, config: dict = None):
#         """
#         Initialize the async video processor with configuration
        
#         Args:
#             config (dict): Configuration parameters
#         """
#         self.config = config or {}
        
#         # Validate required environment variables
#         self._validate_environment()
        
#         # Initialize components
#         self.transcriber = self._initialize_transcriber()
#         self.summarizer = self._initialize_summarizer()
    
#     def _validate_environment(self):
#         """Validate that required environment variables are set"""
#         required_vars = [
#             'AZURE_OPENAI_API_KEY',
#             'AZURE_OPENAI_ENDPOINT', 
#             'AZURE_OPENAI_API_VERSION',
#             'AZURE_OPENAI_DEPLOYMENT_NAME'
#         ]
        
#         missing_vars = [var for var in required_vars if not os.getenv(var)]
#         if missing_vars:
#             raise ValueError(f"Missing required environment variables: {missing_vars}")
        
#         # Check TTS engine specific requirements
#         tts_engine = self.config.get('tts_engine', 'gtts').lower()
#         if tts_engine == 'murf' and not os.getenv('MURF_API_KEY'):
#             raise ValueError("MURF_API_KEY environment variable required when using Murf TTS")
        
#         # Check transcription method specific requirements
#         transcription_method = self.config.get('transcription_method', 'local')
#         if transcription_method == 'api' and not os.getenv('LEMON_FOX_API_KEY'):
#             raise ValueError("LEMON_FOX_API_KEY environment variable required when using API transcription")
    
#     def _initialize_transcriber(self) -> VideoTranscriber:
#         """Initialize the video transcriber with configuration"""
#         transcription_method = self.config.get('transcription_method', 'local')
#         model_size = self.config.get('whisper_model_size', 'small')
#         api_key = os.getenv('LEMON_FOX_API_KEY') if transcription_method == 'api' else None
#         language = self.config.get('transcription_language', 'en')
        
#         return VideoTranscriber(
#             transcription_method=transcription_method,
#             model_size=model_size,
#             api_key=api_key,
#             language=language
#         )
    
#     def _initialize_summarizer(self) -> VideoSummarizer:
#         """Initialize the video summarizer with configuration"""
#         azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
#         azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
#         azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
#         azure_deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        
#         murf_api_key = os.getenv('MURF_API_KEY')
#         tts_engine = self.config.get('tts_engine', 'gtts')
#         target_wpm = self.config.get('target_wpm', 160.0)
        
#         return VideoSummarizer(
#             azure_openai_key=azure_openai_key,
#             azure_openai_endpoint=azure_openai_endpoint,
#             azure_api_version=azure_api_version,
#             azure_deployment_name=azure_deployment_name,
#             murf_api_key=murf_api_key,
#             tts_engine=tts_engine,
#             target_wpm=target_wpm
#         )
    
#     async def process_video_async(self, input_video_path: str, output_video_path: str = None, 
#                                  keep_intermediate_files: bool = False) -> dict:
#         """
#         Main async method to process video from input to final optimized output
        
#         Args:
#             input_video_path (str): Path to input video file
#             output_video_path (str): Path for output video (optional)
#             keep_intermediate_files (bool): Whether to keep temporary files
            
#         Returns:
#             dict: Processing results including paths and cost information
#         """
#         if not os.path.exists(input_video_path):
#             raise FileNotFoundError(f"Input video file not found: {input_video_path}")
        
#         # Generate unique session ID for this request
#         import uuid
#         session_id = str(uuid.uuid4())[:8]
        
#         # Generate output path if not provided
#         if output_video_path is None:
#             input_name = Path(input_video_path).stem
#             output_video_path = f"{input_name}_optimized_{session_id}.mp4"
        
#         logger.info(f"Starting async video processing pipeline for session {session_id}")
#         logger.info(f"Input: {input_video_path}")
#         logger.info(f"Output: {output_video_path}")
        
#         # Use temporary directory for all processing
#         with tempfile.TemporaryDirectory() as temp_dir:
#             try:
#                 # Step 1: Transcribe video to SRT
#                 logger.info("Step 1: Transcribing video to SRT")
#                 language, srt_content = await self.transcriber.transcribe_video_async(input_video_path)
                
#                 if not srt_content or not srt_content.strip():
#                     raise ValueError("Transcription failed - no SRT content generated")
                
#                 logger.info(f"Transcription complete. Language: {language}, Content length: {len(srt_content)} chars")
                
#                 # Optional: Save SRT content for debugging
#                 srt_debug_path = None
#                 if keep_intermediate_files:
#                     srt_debug_path = os.path.join(temp_dir, f"transcription_{session_id}.srt")
#                     with open(srt_debug_path, 'w', encoding='utf-8') as f:
#                         f.write(srt_content)
#                     logger.info(f"Saved SRT content to: {srt_debug_path}")
                    
#                     # Copy SRT to final location for keeping
#                     final_srt_path = output_video_path.replace('.mp4', '.srt')
#                     shutil.copy2(srt_debug_path, final_srt_path)
#                     srt_debug_path = final_srt_path
                
#                 # Step 2: Process SRT and create optimized video
#                 logger.info("Step 2: Processing SRT and creating optimized video")
#                 temp_output_path, cost_summary = await self.summarizer.process_srt_content_to_video_async(
#                     srt_content, input_video_path
#                 )
                
#                 # Step 3: Move final output to desired location
#                 logger.info("Step 3: Moving final output to destination")
#                 if os.path.exists(temp_output_path):
#                     # Copy to final destination
#                     shutil.copy2(temp_output_path, output_video_path)
#                     logger.info(f"Final video saved to: {output_video_path}")
#                 else:
#                     raise FileNotFoundError(f"Expected output file not found: {temp_output_path}")
                
#                 # Prepare results
#                 results = {
#                     "success": True,
#                     "session_id": session_id,
#                     "input_video": input_video_path,
#                     "output_video": output_video_path,
#                     "transcription": {
#                         "language": language,
#                         "method": self.transcriber.transcription_method,
#                         "content_length": len(srt_content),
#                         "srt_file": srt_debug_path if keep_intermediate_files else None
#                     },
#                     "optimization": {
#                         "segments_processed": len(self.summarizer.parse_srt_content(srt_content)),
#                         "tts_engine": self.summarizer.tts_engine,
#                         "target_wpm": self.summarizer.target_wpm
#                     },
#                     "cost_summary": cost_summary
#                 }
                
#                 logger.info("Async video processing pipeline completed successfully")
                
#                 # Log cost summary
#                 logger.info("--- COST & USAGE SUMMARY ---")
#                 logger.info(f"Total Prompt Tokens: {cost_summary['total_prompt_tokens']}")
#                 logger.info(f"Total Completion Tokens: {cost_summary['total_completion_tokens']}")
#                 logger.info(f"Total Tokens Used: {cost_summary['total_tokens']}")
#                 logger.info(f"Estimated OpenAI Cost: ${cost_summary['estimated_cost_usd']:.6f}")
                
#                 return results
                
#             except Exception as e:
#                 logger.error(f"Error in async video processing pipeline: {e}")
#                 error_results = {
#                     "success": False,
#                     "session_id": session_id,
#                     "error": str(e),
#                     "input_video": input_video_path
#                 }
#                 return error_results

# # Global processor instance for handling multiple concurrent requests
# _processor_instance = None
# _processor_lock = asyncio.Lock()

# async def get_processor_instance(config: dict = None) -> AsyncVideoProcessor:
#     """Get or create a singleton processor instance (thread-safe)"""
#     global _processor_instance, _processor_lock
    
#     async with _processor_lock:
#         if _processor_instance is None:
#             _processor_instance = AsyncVideoProcessor(config)
#         return _processor_instance

# async def process_video_request(input_video_path: str, output_video_path: str = None, 
#                                keep_intermediate_files: bool = False, config: dict = None) -> dict:
#     """
#     High-level async function to process a video request
#     This function can handle multiple concurrent requests safely
    
#     Args:
#         input_video_path (str): Path to input video file
#         output_video_path (str): Path for output video (optional)
#         keep_intermediate_files (bool): Whether to keep temporary files
#         config (dict): Configuration for this specific request
        
#     Returns:
#         dict: Processing results
#     """
#     processor = await get_processor_instance(config)
#     return await processor.process_video_async(input_video_path, output_video_path, keep_intermediate_files)

# async def process_multiple_videos_concurrently(video_requests: list, max_concurrent: int = 3) -> list:
#     """
#     Process multiple video requests concurrently with a limit on concurrent processing
    
#     Args:
#         video_requests (list): List of dicts with 'input_path', 'output_path', 'config', etc.
#         max_concurrent (int): Maximum number of concurrent video processing tasks
        
#     Returns:
#         list: Results for each video request
#     """
#     semaphore = asyncio.Semaphore(max_concurrent)
    
#     async def process_single_request(request):
#         async with semaphore:
#             return await process_video_request(
#                 input_video_path=request['input_path'],
#                 output_video_path=request.get('output_path'),
#                 keep_intermediate_files=request.get('keep_intermediate_files', False),
#                 config=request.get('config')
#             )
    
#     tasks = [process_single_request(request) for request in video_requests]
#     results = await asyncio.gather(*tasks, return_exceptions=True)
    
#     return results

# async def main_async():
#     """Main async execution function with configuration"""
    
#     # === CONFIGURATION SECTION ===
#     CONFIG = {
#         # Transcription settings
#         'transcription_method': 'local',  # 'local' or 'api'
#         'whisper_model_size': 'small',    # 'tiny', 'base', 'small', 'medium', 'large'
#         'transcription_language': 'en',   # Language code for transcription
        
#         # TTS settings
#         'tts_engine': 'gtts',            # 'gtts' or 'murf'
#         'target_wpm': 160.0,             # Target words per minute for optimization
#     }
    
#     # Input/Output paths
#     INPUT_VIDEO = r"C:\Users\kisha\Downloads\windowsinstallation.mp4"
#     OUTPUT_VIDEO = "optimized_windows_installation.mp4"  # Optional, will auto-generate if None
    
#     # Processing options
#     KEEP_INTERMEDIATE_FILES = False  # Set to True for debugging
    
#     # === END CONFIGURATION ===
    
#     try:
#         # Process single video
#         results = await process_video_request(
#             input_video_path=INPUT_VIDEO,
#             output_video_path=OUTPUT_VIDEO,
#             keep_intermediate_files=KEEP_INTERMEDIATE_FILES,
#             config=CONFIG
#         )
        
#         # Print results
#         if results["success"]:
#             print("\n" + "="*50)
#             print("VIDEO PROCESSING COMPLETED SUCCESSFULLY!")
#             print("="*50)
#             print(f"Session ID: {results['session_id']}")
#             print(f"Input Video: {results['input_video']}")
#             print(f"Output Video: {results['output_video']}")
#             print(f"Transcription Language: {results['transcription']['language']}")
#             print(f"Transcription Method: {results['transcription']['method']}")
#             print(f"Segments Processed: {results['optimization']['segments_processed']}")
#             print(f"TTS Engine: {results['optimization']['tts_engine']}")
#             print(f"Estimated Cost: ${results['cost_summary']['estimated_cost_usd']:.6f}")
            
#             if KEEP_INTERMEDIATE_FILES and results['transcription']['srt_file']:
#                 print(f"SRT file saved to: {results['transcription']['srt_file']}")
        
#         else:
#             print("\n" + "="*50)
#             print("VIDEO PROCESSING FAILED!")
#             print("="*50)
#             print(f"Session ID: {results['session_id']}")
#             print(f"Error: {results['error']}")
    
#     except Exception as e:
#         logger.error(f"Fatal error in main execution: {e}")
#         print(f"\nFatal error: {e}")
#         sys.exit(1)

# def main():
#     """Synchronous wrapper for async main function"""
#     try:
#         asyncio.run(main_async())
#     except KeyboardInterrupt:
#         print("\nProcess interrupted by user.")
#         sys.exit(0)

# # Example of how to handle multiple videos concurrently
# async def example_multiple_videos():
#     """Example of processing multiple videos concurrently"""
    
#     video_requests = [
#         {
#             'input_path': 'video1.mp4',
#             'output_path': 'output1.mp4',
#             'config': {'tts_engine': 'gtts', 'target_wpm': 150}
#         },
#         {
#             'input_path': 'video2.mp4',
#             'output_path': 'output2.mp4', 
#             'config': {'tts_engine': 'murf', 'target_wpm': 170}
#         },
#         {
#             'input_path': 'video3.mp4',
#             'config': {'transcription_method': 'api'}  # Will auto-generate output path
#         }
#     ]
    
#     # Process up to 2 videos concurrently
#     results = await process_multiple_videos_concurrently(video_requests, max_concurrent=2)
    
#     for i, result in enumerate(results):
#         if isinstance(result, Exception):
#             print(f"Video {i+1} failed: {result}")
#         elif result.get('success'):
#             print(f"Video {i+1} completed: {result['output_video']}")
#         else:
#             print(f"Video {i+1} failed: {result.get('error', 'Unknown error')}")

# if __name__ == "__main__":
#     main()

import os
import sys
import logging
import tempfile
import shutil
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv
from video_transcriber import VideoTranscriber
from video_summarizer import VideoSummarizer
from google_drive_uploader import GoogleDriveUploader
from mongodb_manager import MongoDBManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('video_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class AsyncVideoProcessor:
    """Async main orchestrator for video transcription, summarization, and cloud storage pipeline"""
    
    def __init__(self, config: dict = None):
        """
        Initialize the async video processor with configuration
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {}
        
        # Validate required environment variables
        self._validate_environment()
        
        # Initialize components
        self.transcriber = self._initialize_transcriber()
        self.summarizer = self._initialize_summarizer()
        self.gdrive_uploader = self._initialize_gdrive_uploader()
        self.mongodb_manager = self._initialize_mongodb_manager()
    
    def _validate_environment(self):
        """Validate that required environment variables are set"""
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT', 
            'AZURE_OPENAI_API_VERSION',
            'AZURE_OPENAI_DEPLOYMENT_NAME',
            'GOOGLE_SERVICE_ACCOUNT_PATH',
            'MONGODB_URI'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Check TTS engine specific requirements
        tts_engine = self.config.get('tts_engine', 'gtts').lower()
        if tts_engine == 'murf' and not os.getenv('MURF_API_KEY'):
            raise ValueError("MURF_API_KEY environment variable required when using Murf TTS")
        
        # Check transcription method specific requirements
        transcription_method = self.config.get('transcription_method', 'local')
        if transcription_method == 'api' and not os.getenv('LEMON_FOX_API_KEY'):
            raise ValueError("LEMON_FOX_API_KEY environment variable required when using API transcription")
    
    def _initialize_transcriber(self) -> VideoTranscriber:
        """Initialize the video transcriber with configuration"""
        transcription_method = self.config.get('transcription_method', 'local')
        model_size = self.config.get('whisper_model_size', 'small')
        api_key = os.getenv('LEMON_FOX_API_KEY') if transcription_method == 'api' else None
        language = self.config.get('transcription_language', 'en')
        
        return VideoTranscriber(
            transcription_method=transcription_method,
            model_size=model_size,
            api_key=api_key,
            language=language
        )
    
    def _initialize_summarizer(self) -> VideoSummarizer:
        """Initialize the video summarizer with configuration"""
        azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        azure_deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        
        murf_api_key = os.getenv('MURF_API_KEY')
        tts_engine = self.config.get('tts_engine', 'gtts')
        target_wpm = self.config.get('target_wpm', 160.0)
        
        return VideoSummarizer(
            azure_openai_key=azure_openai_key,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment_name=azure_deployment_name,
            murf_api_key=murf_api_key,
            tts_engine=tts_engine,
            target_wpm=target_wpm
        )
    
    # def _initialize_gdrive_uploader(self) -> GoogleDriveUploader:
    #     """Initialize the Google Drive uploader"""
    #     service_account_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_PATH')
    #     folder_id = self.config.get('gdrive_folder_id') or os.getenv('GDRIVE_FOLDER_ID')
        
    #     return GoogleDriveUploader(
    #         service_account_path=service_account_path,
    #         folder_id=folder_id
    #     )

    def _initialize_gdrive_uploader(self) -> GoogleDriveUploader:
        """Initialize the Google Drive uploader"""
        service_account_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_PATH')
        folder_id = self.config.get('gdrive_folder_id') or os.getenv('GDRIVE_FOLDER_ID')
        
        # Log the folder configuration for debugging
        if folder_id:
            logger.info(f"Google Drive folder ID configured: {folder_id}")
        else:
            logger.info("No Google Drive folder ID specified - will upload to root directory")
        
        return GoogleDriveUploader(
            service_account_path=service_account_path,
            folder_id=folder_id
        )    
    
    def _initialize_mongodb_manager(self) -> MongoDBManager:
        """Initialize the MongoDB manager"""
        mongodb_uri = os.getenv('MONGODB_URI')
        database_name = self.config.get('mongodb_database', 'emoai2')
        
        return MongoDBManager(
            mongodb_uri=mongodb_uri,
            database_name=database_name
        )
    
    async def process_video_async(self, 
                                 input_video_path: str, 
                                 user_id: str,
                                 output_video_path: str = None, 
                                 keep_intermediate_files: bool = False) -> dict:
        """
        Main async method to process video from input to final optimized output with cloud storage
        
        Args:
            input_video_path (str): Path to input video file
            user_id (str): User identifier for database storage
            output_video_path (str): Path for output video (optional, used for temp processing)
            keep_intermediate_files (bool): Whether to keep temporary files locally
            
        Returns:
            dict: Processing results including cloud storage information
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video file not found: {input_video_path}")
        
        # Generate unique session ID for this request
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        start_time = time.time()
        
        # Generate output path if not provided
        if output_video_path is None:
            input_name = Path(input_video_path).stem
            output_video_path = f"{input_name}_optimized_{session_id}.mp4"
        
        logger.info(f"Starting async video processing pipeline for session {session_id}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Input: {input_video_path}")
        logger.info(f"Temp Output: {output_video_path}")
        
        # Use temporary directory for all processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_output_path = None
            try:
                # Step 1: Transcribe video to SRT
                logger.info("Step 1: Transcribing video to SRT")
                language, srt_content = await self.transcriber.transcribe_video_async(input_video_path)
                
                if not srt_content or not srt_content.strip():
                    raise ValueError("Transcription failed - no SRT content generated")
                
                logger.info(f"Transcription complete. Language: {language}, Content length: {len(srt_content)} chars")
                
                # Optional: Save SRT content for debugging
                srt_debug_path = None
                if keep_intermediate_files:
                    srt_debug_path = os.path.join(temp_dir, f"transcription_{session_id}.srt")
                    with open(srt_debug_path, 'w', encoding='utf-8') as f:
                        f.write(srt_content)
                    logger.info(f"Saved SRT content to: {srt_debug_path}")
                
                # Step 2: Process SRT and create optimized video
                logger.info("Step 2: Processing SRT and creating optimized video")
                temp_output_path, cost_summary = await self.summarizer.process_srt_content_to_video_async(
                    srt_content, input_video_path
                )
                
                # Step 3: Copy to local output path temporarily
                logger.info("Step 3: Preparing video for upload")
                if os.path.exists(temp_output_path):
                    # Copy to specified output location temporarily
                    shutil.copy2(temp_output_path, output_video_path)
                    logger.info(f"Temp video saved to: {output_video_path}")
                else:
                    raise FileNotFoundError(f"Expected output file not found: {temp_output_path}")
                
                # Step 4: Upload to Google Drive
                logger.info("Step 4: Uploading video to Google Drive")
                upload_filename = f"{Path(input_video_path).stem}_optimized_{session_id}.mp4"
                gdrive_result = await self.gdrive_uploader.upload_video_async(
                    output_video_path, 
                    upload_filename
                )
                
                if not gdrive_result['success']:
                    raise Exception(f"Google Drive upload failed: {gdrive_result['error']}")
                
                logger.info("Google Drive upload successful!")
                logger.info(f"File ID: {gdrive_result['file_id']}")
                logger.info(f"View Link: {gdrive_result['web_view_link']}")
                
                # Step 5: Store record in MongoDB
                logger.info("Step 5: Storing record in MongoDB")

                async with self.mongodb_manager as mongo:
                
                    processing_time = time.time() - start_time
                    processing_data = {
                        "input_video": input_video_path,
                        "output_video": output_video_path,
                        "transcription": {
                            "language": language,
                            "method": self.transcriber.transcription_method,
                            "content_length": len(srt_content)
                        },
                        "optimization": {
                            "segments_processed": len(self.summarizer.parse_srt_content(srt_content)),
                            "tts_engine": self.summarizer.tts_engine,
                            "target_wpm": self.summarizer.target_wpm
                        },
                        "cost_summary": cost_summary,
                        "processing_time": processing_time
                    }
                    
                    mongo_result = await self.mongodb_manager.store_video_record(
                        user_id=user_id,
                        session_id=session_id,
                        gdrive_data=gdrive_result,
                        processing_data=processing_data
                    )
                    
                    if not mongo_result['success']:
                        logger.warning(f"MongoDB storage failed: {mongo_result['error']}")
                        # Continue execution even if MongoDB fails
                
                
                # Step 6: Delete local video file after successful upload and storage
                logger.info("Step 6: Cleaning up local files")
                try:
                    if os.path.exists(output_video_path):
                        os.remove(output_video_path)
                        logger.info(f"Deleted local video file: {output_video_path}")
                except Exception as e:
                    logger.warning(f"Could not delete local video file: {e}")
                
                # Keep SRT file if requested
                final_srt_path = None
                if keep_intermediate_files and srt_debug_path:
                    final_srt_path = output_video_path.replace('.mp4', '.srt')
                    shutil.copy2(srt_debug_path, final_srt_path)
                    logger.info(f"Kept SRT file: {final_srt_path}")
                
                # Prepare results
                results = {
                    "success": True,
                    "session_id": session_id,
                    "user_id": user_id,
                    "input_video": input_video_path,
                    "local_output_deleted": True,
                    "processing_time": processing_time,
                    
                    # Google Drive information
                    "gdrive": gdrive_result,
                    
                    # MongoDB information
                    "mongodb": mongo_result,
                    
                    # Processing details
                    "transcription": {
                        "language": language,
                        "method": self.transcriber.transcription_method,
                        "content_length": len(srt_content),
                        "srt_file": final_srt_path if keep_intermediate_files else None
                    },
                    "optimization": {
                        "segments_processed": len(self.summarizer.parse_srt_content(srt_content)),
                        "tts_engine": self.summarizer.tts_engine,
                        "target_wpm": self.summarizer.target_wpm
                    },
                    "cost_summary": cost_summary
                }
                
                logger.info("Async video processing pipeline completed successfully")
                
                # Log comprehensive summary
                logger.info("--- PROCESSING SUMMARY ---")
                logger.info(f"Session ID: {session_id}")
                logger.info(f"User ID: {user_id}")
                logger.info(f"Processing Time: {processing_time:.2f} seconds")
                logger.info(f"Google Drive File ID: {gdrive_result['file_id']}")
                logger.info(f"MongoDB Document ID: {mongo_result.get('document_id', 'N/A')}")
                logger.info(f"Estimated OpenAI Cost: ${cost_summary['estimated_cost_usd']:.6f}")
                logger.info("Local video file deleted after upload")
                
                return results
                
            except Exception as e:
                logger.error(f"Error in async video processing pipeline: {e}")
                
                # Cleanup local file on error
                if temp_output_path and os.path.exists(output_video_path):
                    try:
                        os.remove(output_video_path)
                        logger.info("Cleaned up local video file after error")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not cleanup local file: {cleanup_error}")
                
                error_results = {
                    "success": False,
                    "session_id": session_id,
                    "user_id": user_id,
                    "error": str(e),
                    "input_video": input_video_path,
                    "processing_time": time.time() - start_time
                }
                return error_results

# Global processor instance for handling multiple concurrent requests
_processor_instance = None
_processor_lock = asyncio.Lock()

async def get_processor_instance(config: dict = None) -> AsyncVideoProcessor:
    """Get or create a singleton processor instance (thread-safe)"""
    global _processor_instance, _processor_lock
    
    async with _processor_lock:
        if _processor_instance is None:
            _processor_instance = AsyncVideoProcessor(config)
            # Initialize MongoDB connection
            await _processor_instance.mongodb_manager.initialize()
        return _processor_instance

async def process_video_request(input_video_path: str, 
                               user_id: str,
                               output_video_path: str = None, 
                               keep_intermediate_files: bool = False, 
                               config: dict = None) -> dict:
    """
    High-level async function to process a video request with cloud storage
    This function can handle multiple concurrent requests safely
    
    Args:
        input_video_path (str): Path to input video file
        user_id (str): User identifier for database storage
        output_video_path (str): Path for temp output video (optional)
        keep_intermediate_files (bool): Whether to keep temporary files
        config (dict): Configuration for this specific request
        
    Returns:
        dict: Processing results including cloud storage information
    """
    processor = await get_processor_instance(config)
    return await processor.process_video_async(
        input_video_path, user_id, output_video_path, keep_intermediate_files
    )

async def process_multiple_videos_concurrently(video_requests: list, max_concurrent: int = 3) -> list:
    """
    Process multiple video requests concurrently with a limit on concurrent processing
    
    Args:
        video_requests (list): List of dicts with 'input_path', 'user_id', 'output_path', 'config', etc.
        max_concurrent (int): Maximum number of concurrent video processing tasks
        
    Returns:
        list: Results for each video request
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_request(request):
        async with semaphore:
            return await process_video_request(
                input_video_path=request['input_path'],
                user_id=request['user_id'],
                output_video_path=request.get('output_path'),
                keep_intermediate_files=request.get('keep_intermediate_files', False),
                config=request.get('config')
            )
    
    tasks = [process_single_request(request) for request in video_requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results

# Additional utility functions for database operations
async def get_user_videos(user_id: str, limit: int = 50, offset: int = 0) -> list:
    """Get all videos processed for a specific user"""
    processor = await get_processor_instance()
    return await processor.mongodb_manager.get_user_videos(user_id, limit, offset)

async def get_video_by_session(session_id: str) -> dict:
    """Get video record by session ID"""
    processor = await get_processor_instance()
    return await processor.mongodb_manager.get_video_by_session(session_id)

async def get_processing_stats(user_id: str = None) -> dict:
    """Get processing statistics"""
    processor = await get_processor_instance()
    return await processor.mongodb_manager.get_processing_stats(user_id)

async def main_async():
    """Main async execution function with configuration"""
    
    # === CONFIGURATION SECTION ===
    CONFIG = {
        # Transcription settings
        'transcription_method': 'local',  # 'local' or 'api'
        'whisper_model_size': 'small',    # 'tiny', 'base', 'small', 'medium', 'large'
        'transcription_language': 'en',   # Language code for transcription
        
        # TTS settings
        'tts_engine': 'gtts',            # 'gtts' or 'murf'
        'target_wpm': 160.0,             # Target words per minute for optimization
        
        # Cloud storage settings
        'gdrive_folder_id': None,        # Optional: specific Google Drive folder
        'mongodb_database': 'emoai2'  # MongoDB database name
    }
    
    # Input/Output paths and user info
    INPUT_VIDEO = r"C:\Users\kisha\Downloads\windowsinstallation.mp4"
    USER_ID = "user_12345"  # Replace with actual user ID
    OUTPUT_VIDEO = None  # Will auto-generate, used for temp processing only
    
    # Processing options
    KEEP_INTERMEDIATE_FILES = False  # Set to True to keep SRT files
    
    # === END CONFIGURATION ===
    
    try:
        # Process single video
        results = await process_video_request(
            input_video_path=INPUT_VIDEO,
            user_id=USER_ID,
            output_video_path=OUTPUT_VIDEO,
            keep_intermediate_files=KEEP_INTERMEDIATE_FILES,
            config=CONFIG
        )
        
        # Print results
        if results["success"]:
            print("\n" + "="*60)
            print("VIDEO PROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Session ID: {results['session_id']}")
            print(f"User ID: {results['user_id']}")
            print(f"Processing Time: {results['processing_time']:.2f} seconds")
            print(f"Input Video: {results['input_video']}")
            print(f"Local File Deleted: {results['local_output_deleted']}")
            
            print(f"\n--- GOOGLE DRIVE INFO ---")
            gdrive = results['gdrive']
            print(f"File ID: {gdrive['file_id']}")
            print(f"File Name: {gdrive['file_name']}")
            print(f"View Link: {gdrive['web_view_link']}")
            print(f"Download Link: {gdrive['download_link']}")
            print(f"File Size: {gdrive.get('size', 0)} bytes")
            
            print(f"\n--- DATABASE INFO ---")
            mongodb = results['mongodb']
            print(f"MongoDB Success: {mongodb['success']}")
            print(f"Document ID: {mongodb.get('document_id', 'N/A')}")
            
            print(f"\n--- PROCESSING DETAILS ---")
            print(f"Language: {results['transcription']['language']}")
            print(f"Transcription Method: {results['transcription']['method']}")
            print(f"Segments Processed: {results['optimization']['segments_processed']}")
            print(f"TTS Engine: {results['optimization']['tts_engine']}")
            print(f"Estimated Cost: ${results['cost_summary']['estimated_cost_usd']:.6f}")
            
            if KEEP_INTERMEDIATE_FILES and results['transcription']['srt_file']:
                print(f"SRT File: {results['transcription']['srt_file']}")
        
        else:
            print("\n" + "="*50)
            print("VIDEO PROCESSING FAILED!")
            print("="*50)
            print(f"Session ID: {results['session_id']}")
            print(f"User ID: {results['user_id']}")
            print(f"Error: {results['error']}")
            print(f"Processing Time: {results['processing_time']:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)

def main():
    """Synchronous wrapper for async main function"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)

# Example of how to handle multiple videos with different users
async def example_multiple_videos():
    """Example of processing multiple videos concurrently for different users"""
    
    video_requests = [
        {
            'input_path': 'video1.mp4',
            'user_id': 'user_001',
            'config': {'tts_engine': 'gtts', 'target_wpm': 150}
        },
        {
            'input_path': 'video2.mp4',
            'user_id': 'user_002',
            'config': {'tts_engine': 'murf', 'target_wpm': 170}
        },
        {
            'input_path': 'video3.mp4',
            'user_id': 'user_003',
            'config': {'transcription_method': 'api'}
        }
    ]
    
    # Process up to 2 videos concurrently
    results = await process_multiple_videos_concurrently(video_requests, max_concurrent=2)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Video {i+1} failed: {result}")
        elif result.get('success'):
            print(f"Video {i+1} completed successfully:")
            print(f"  - Session ID: {result['session_id']}")
            print(f"  - User ID: {result['user_id']}")
            print(f"  - Google Drive File ID: {result['gdrive']['file_id']}")
            print(f"  - View Link: {result['gdrive']['web_view_link']}")
        else:
            print(f"Video {i+1} failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()