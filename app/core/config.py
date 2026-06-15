"""
app/core/config.py
──────────────────
Centralized application settings using Pydantic BaseSettings.

WHY: Instead of scattered os.getenv() calls throughout the codebase,
all config lives here. Change a setting once → updates everywhere.
This follows the 12-Factor App methodology used in production systems.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI-Powered Customer Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = True

    model_path: str = "model/kmeans_model.pkl"
    scaler_path: str = "model/scaler.pkl"
    label_encoder_path: str = "model/label_encoder.pkl"
    cluster_labels_path: str = "model/cluster_labels.json"

    openai_api_key: str = ""

    # Database configuration (Phase 2)
    database_url: str = "sqlite:///./customer_intelligence.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
