# app/services/video_processor.py
import os
import logging
from app.config import get_settings

# Import your existing modules - update these import paths to match your file structure
from app.services.video_transcriber import VideoTranscriber
from app.services.video_summarizer2 import VideoSummarizer, SRTSegment, OptimizedSegment

logger = logging.getLogger(__name__)

class AsyncVideoProcessor:
    """Async main orchestrator for video transcription and summarization"""
    
    def __init__(self, config: dict = None):
        """Initialize the async video processor with configuration"""
        self.config = config or {}
        settings = get_settings()
        
        # Validate required environment variables
        self._validate_environment()
        
        # Initialize transcriber
        self.transcriber = self._initialize_transcriber()
        
        # Initialize summarizer  
        self.summarizer = self._initialize_summarizer()
    
    def _validate_environment(self):
        """Validate that required environment variables are set"""
        settings = get_settings()
        
        required_vars = [
            'azure_openai_api_key',
            'azure_openai_endpoint', 
            'azure_openai_api_version',
            'azure_openai_deployment_name',
            'google_service_account_path',
            'mongodb_uri'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(settings, var, None):
                missing_vars.append(var.upper())
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Check TTS engine specific requirements
        tts_engine = self.config.get('tts_engine', settings.tts_engine).lower()
        if tts_engine == 'murf' and not settings.murf_api_key:
            raise ValueError("MURF_API_KEY environment variable required when using Murf TTS")
        
        # Check transcription method specific requirements
        transcription_method = self.config.get('transcription_method', settings.transcription_method)
        if transcription_method == 'api' and not settings.lemon_fox_api_key:
            raise ValueError("LEMON_FOX_API_KEY environment variable required when using API transcription")
    
    def _initialize_transcriber(self) -> VideoTranscriber:
        """Initialize the video transcriber with configuration"""
        settings = get_settings()
        
        transcription_method = self.config.get('transcription_method', settings.transcription_method)
        model_size = self.config.get('whisper_model_size', settings.whisper_model_size)
        api_key = settings.lemon_fox_api_key if transcription_method == 'api' else None
        language = self.config.get('transcription_language', settings.transcription_language)
        
        return VideoTranscriber(
            transcription_method=transcription_method,
            model_size=model_size,
            api_key=api_key,
            language=language
        )
    
    def _initialize_summarizer(self) -> VideoSummarizer:
        """Initialize the video summarizer with configuration"""
        settings = get_settings()
        
        murf_api_key = settings.murf_api_key
        tts_engine = self.config.get('tts_engine', settings.tts_engine)
        target_wpm = self.config.get('target_wpm', settings.target_wpm)
        
        return VideoSummarizer(
            azure_openai_key=settings.azure_openai_api_key,
            azure_openai_endpoint=settings.azure_openai_endpoint,
            azure_api_version=settings.azure_openai_api_version,
            azure_deployment_name=settings.azure_openai_deployment_name,
            murf_api_key=murf_api_key,
            tts_engine=tts_engine,
            target_wpm=target_wpm
        )