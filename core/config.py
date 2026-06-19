from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str

    LANGSMITH_API_KEY: str
    LANGCHAIN_PROJECT: str
    LANGCHAIN_TRACING_V2: bool

    BIG_MODEL: str
    SMALL_MODEL: str

    ENV: Literal["dev", "prod"]
    
    LOG_DIR: str
    LOG_LEVEL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
