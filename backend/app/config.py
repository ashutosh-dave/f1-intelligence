"""
Backend configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    app_name: str = "F1 Race Intelligence Engine"
    app_version: str = "1.0.0"
    debug: bool = True

    # Database
    database_url: str = "sqlite:///./f1_intelligence.db"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Simulation
    default_simulations: int = 5000
    max_simulations: int = 20000

    # ML
    ml_weight: float = 0.4
    sim_weight: float = 0.6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
