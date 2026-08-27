"""Application configuration, loaded from environment variables."""

import json
from functools import lru_cache
from typing import Annotated, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Kaya API"
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    # --- Database ---
    mongodb_uri: str = Field(default="")
    mongodb_db_name: str = Field(default="KayaDB")

    # --- Auth ---
    # Override in production. Generate one with:  openssl rand -hex 32
    jwt_secret: str = Field(default="dev-only-insecure-secret-change-me")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # --- LLM providers ---
    groq_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    llm_temperature: float = Field(default=0.4)
    # Caps every per-schema output budget. Leave unset to use the defaults in
    # `app/agent/llm.py`; set it when your key's tokens-per-minute limit is
    # tighter than those assume, which shows up as a 413 rather than a 429.
    llm_max_tokens: Optional[int] = Field(default=None, ge=256, le=32000)

    @field_validator("llm_max_tokens", mode="before")
    @classmethod
    def _blank_means_unset(cls, v):
        """`LLM_MAX_TOKENS=` with nothing after it means "use the defaults".

        Leaving an optional key blank is the obvious thing to do, and pydantic
        would otherwise refuse to parse "" as an int and take the whole app
        down at import time over a setting nobody needed.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    # --- CORS ---
    # `NoDecode` is load-bearing. Without it pydantic-settings sees a list-typed
    # field and runs `json.loads` on the raw environment value *inside the env
    # source*, before any validator gets a look — so the comma-separated form
    # every deployment doc reaches for dies with "Expecting value: line 1
    # column 1" and the app never starts.
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        """Accept CORS_ORIGINS as comma-separated or as a JSON array.

        Comma-separated is what the docs tell you to paste into Render. JSON is
        what a platform that round-trips config through JSON may hand back, and
        what the old list-typed field used to require — so both keep working.
        """
        if not isinstance(v, str):
            return v

        text = v.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass  # fall through and treat it as comma-separated
        return [origin.strip() for origin in text.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — read the environment exactly once."""
    return Settings()


settings = get_settings()
