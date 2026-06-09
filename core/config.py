from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    PROJECT_NAME: str = "AI Coding Mentor"
    VERSION: str = "1.0.0"
    DEBUG_MODE: bool = True
    NOVITA_API_KEY: str
    LANGSMITH_API_KEY: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()