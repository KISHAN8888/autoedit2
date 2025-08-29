# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Environment
    environment: str = "development"
    
    # MongoDB
    mongodb_uri: str
    mongodb_database: str = "emoai2"
    
    # Azure OpenAI
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str
    azure_openai_deployment_name: str
    
    # TTS
    murf_api_key: Optional[str] = None
    tts_engine: str = "gtts"
    target_wpm: float = 160.0
    
    # Google Drive
    google_service_account_path: str
    gdrive_folder_id: Optional[str] = None
    
    # Transcription
    transcription_method: str = "api"
    whisper_model_size: str = "small"
    lemon_fox_api_key: Optional[str] = "5cCrTnA7Nv73iujYaIxW302WLbfuVCnR"
    transcription_language: str = "en"
    
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