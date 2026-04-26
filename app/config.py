"""Application configuration using pydantic-settings.

Reads from environment variables with fallback to a .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    MODEL_PATH: str = "./model_v5"
    AWS_REGION: str = "us-east-1"
    AWS_PROFILE: str = ""
    ADMIN_PASSWORD: str = "admin"
    BEDROCK_MODEL_ID: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    SEVERITY_THRESHOLD: float = 0.45
    MAX_DIAGNOSES_SHOWN: int = 3 
    MAX_INTERVENTIONS_SHOWN: int = 3
    FRONTEND_ORIGIN: str = "http://localhost:3000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
