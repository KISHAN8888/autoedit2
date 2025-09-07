# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    
    # MongoDB
    mongodb_uri: str
    mongodb_database: str = "emoai2"
    
    # JWT Authentication
    jwt_secret_key: str 
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    
    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str
    azure_openai_deployment_name: str
    
    # TTS
    murf_api_key: Optional[str] = None
    tts_engine: str = "murf" # vibe_voice, groq_playai, murf
    target_wpm: float = 160.0

    # vibe-voice-1.5B -tts
    vibe_voice_api_url: Optional[str] = None
    vibe_voice_speaker_names: str = None

    # playai-groq -tts
    groq_playai_speaker_name: str = None    
    # Google Drive
    google_service_account_path: str
    gdrive_folder_id: Optional[str] = None
    
    # Transcription
    transcription_method: str = "api" # api, local, groq
    whisper_model_size: str = "small"
    lemon_fox_api_key: Optional[str]
    transcription_language: str = "en"

    groq_api_key: Optional[str]
    
    # Celery
    celery_broker_url: str ="redis://localhost:6379/0"
    celery_result_backend:str ="redis://localhost:6379/0"

    
    # Security
    rate_limit_calls: int = 50
    rate_limit_period: int = 60
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    
    # Processing
    max_concurrent_uploads: int = 10
    worker_concurrency: int = 2

    vibe_voice_config = {
        'api_url': 'https://fyzy94d0jqdy.share.zrok.io/generate-audio/',
        'model_path': 'VibeVoice-1.5B',
        'speaker_name': 'Alice_woman' 
    } # for now vibe voice model is hosted on this url
    # voices for vibe voice
    # ['Alice_woman', 'Carter_man', 'Frank_man', 'Maya_woman', 
    #  'Mary_woman_bgm', 'Samuel_man', 'Anchen_man_bgm', 
    #  'Bowen_man', 'Xiran_woman']

    groq_playai_config = {
        'api_key': groq_api_key, 
        'voice': 'Fritz-PlayAI' 
    }
    # voices for groq-playai
    # (Arista-PlayAI, Atlas-PlayAI, Basil-PlayAI, Briggs-PlayAI, Calum-PlayAI, Celeste-PlayAI, Cheyenne-PlayAI,
    # Chip-PlayAI, Cillian-PlayAI, Deedee-PlayAI, Fritz-PlayAI, Gail-PlayAI, Indigo-PlayAI, Mamaw-PlayAI, 
    # Mason-PlayAI, Mikail-PlayAI, Mitch-PlayAI, Quinn-PlayAI, Thunder-PlayAI)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

# Global settings instance
_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings