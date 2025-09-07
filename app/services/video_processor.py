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
        elif tts_engine == 'vibe_voice' and not settings.vibe_voice_config:
            raise ValueError("VIBE_VOICE_CONFIG environment variable required when using Vibe Voice TTS")
        elif tts_engine == 'groq_playai' and not settings.groq_playai_config:
            raise ValueError("GROQ_PLAYAI_CONFIG environment variable required when using Groq PlayAI TTS")
        
        # Check transcription method specific requirements
        transcription_method = self.config.get('transcription_method', settings.transcription_method)
        if transcription_method == 'api' and not settings.lemon_fox_api_key:
            raise ValueError("LEMON_FOX_API_KEY environment variable required when using API transcription")
        if transcription_method == 'groq' and not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable required when using Groq transcription")
    
    def _initialize_transcriber(self) -> VideoTranscriber:
        """Initialize the video transcriber with configuration"""
        settings = get_settings()
        
        transcription_method = self.config.get('transcription_method', settings.transcription_method)
        model_size = self.config.get('whisper_model_size', settings.whisper_model_size)
        api_key = settings.lemon_fox_api_key if transcription_method == 'api' else None
        language = self.config.get('transcription_language', settings.transcription_language)
        groq_api_key = settings.groq_api_key if transcription_method == 'groq' else None

        return VideoTranscriber(
            transcription_method=transcription_method,
            model_size=model_size,
            api_key=api_key,
            groq_api_key=groq_api_key,
            language=language
        )
    
    def _initialize_summarizer(self) -> VideoSummarizer:
        """Initialize the video summarizer with configuration"""
        settings = get_settings()
        
        murf_api_key = settings.murf_api_key
        tts_engine = self.config.get('tts_engine', settings.tts_engine)
        target_wpm = self.config.get('target_wpm', settings.target_wpm)
        
        # Prepare TTS engine specific configurations
        vibe_voice_config = None
        groq_playai_config = None
        
        if tts_engine == 'vibe_voice':
            vibe_voice_config = {
                'api_url': settings.vibe_voice_config['api_url'],
                'model_path': settings.vibe_voice_config['model_path'],  # Default model path
                'speaker_name': settings.vibe_voice_config['speaker_name'] or 'Alice_woman'  # Default speaker
            }
        elif tts_engine == 'groq_playai':
            groq_playai_config = {
                'api_key': settings.groq_playai_config['api_key'],
                'voice': settings.groq_playai_config['voice'] or 'Fritz-PlayAI'  # Default voice
            }
        
        return VideoSummarizer(
            azure_openai_key=settings.azure_openai_api_key,
            azure_openai_endpoint=settings.azure_openai_endpoint,
            azure_api_version=settings.azure_openai_api_version,
            azure_deployment_name=settings.azure_openai_deployment_name,
            murf_api_key=murf_api_key,
            tts_engine=tts_engine,
            target_wpm=target_wpm,
            vibe_voice_config=vibe_voice_config,
            groq_playai_config=groq_playai_config
        )