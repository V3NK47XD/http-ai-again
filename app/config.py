"""Application configuration settings."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and/or .env file."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq settings
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # Gemini / Google settings (accepts GEMINI_API_KEY or GOOGLE_API_KEY)
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")
    gemini_lite_model: str = Field(default="gemini-3.5-flash-lite", alias="GEMINI_LITE_MODEL")

    # Dedicated model routes
    qwen_model: str = Field(default="qwen/qwen3.8-27b", alias="QWEN_MODEL")
    gpt_model: str = Field(default="openai/gpt-oss-120b", alias="GPT_MODEL")
    gemma_model: str = Field(default="gemma-4-31b-it", alias="GEMMA_MODEL")

    # App server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    system_prompt: str | None = Field(
        default="You are a helpful and concise AI assistant.",
        alias="SYSTEM_PROMPT",
    )

    @property
    def effective_gemini_api_key(self) -> str | None:
        """Returns GEMINI_API_KEY or fallback GOOGLE_API_KEY."""
        return self.gemini_api_key or self.google_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
