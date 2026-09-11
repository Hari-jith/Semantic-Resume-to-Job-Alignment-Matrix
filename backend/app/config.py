"""
Application configuration.

All values are loaded from environment variables (via a .env file in local dev).
Nothing here should ever be hardcoded with a real secret — see .env.example
for the variables this app expects.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Groq API ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # --- Embedding model ---
    finetuned_model_path: str = "models/resume_job_mpnet_finetuned"
    pretrained_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # --- File upload limits ---
    max_file_size_mb: int = 10

    # --- Scoring weights: Resume + JD mode ---
    weight_semantic: float = 0.30
    weight_skills: float = 0.30
    weight_keywords: float = 0.15
    weight_experience: float = 0.15
    weight_completeness: float = 0.10

    # --- Scoring weights: Resume-only mode ---
    weight_resume_completeness: float = 0.35
    weight_resume_skills: float = 0.30
    weight_resume_structure: float = 0.20
    weight_resume_role_signal: float = 0.15

    # --- CORS ---
    frontend_origin: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. Using lru_cache means the .env file is read
    once per process, not on every request.
    """
    return Settings()
