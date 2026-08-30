"""Application configuration, loaded from environment variables."""

import json
from functools import lru_cache
from typing import Annotated, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def normalise_origin(value: str) -> str:
    """Reduce a configured origin to the form a browser actually sends.

    A browser's `Origin` header is scheme://host[:port] and nothing else. No
    path, no trailing slash, lowercase. CORS then compares it to the allow list
    as an exact string, so `https://site.vercel.app/` never matches
    `https://site.vercel.app` and the failure is completely silent: the browser
    reports only that a header was missing.

    That trailing slash is the single most common deployment mistake, and the
    dashboard you paste it into shows you nothing wrong. Since every one of
    these differences is unambiguously a typo. No origin is ever meant to end
    in a slash, or to carry quotes, or to be uppercase. They are corrected
    here rather than left to fail an exact-match comparison at 3am.

    A bare host with no scheme is corrected too. It cannot possibly match, so
    it is certainly a mistake, and https is the only sane reading of one.
    """
    text = value.strip().strip('"').strip("'").strip()
    text = text.rstrip("/")

    if text and "://" not in text:
        text = f"https://{text}"

    # Scheme and host are case-insensitive and browsers always send them
    # lowercased. There is no path to preserve the case of.
    return text.lower()


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
    # Which provider to use. Leave unset to pick whichever key is present,
    # which is what most people want and what the tests rely on.
    llm_provider: Optional[str] = Field(default=None)

    groq_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    # Model names are provider-specific. A Groq name means nothing to Gemini.
    # Leave blank to use the provider's default; `python -m app.tools.check_llm`
    # lists what your key can actually reach.
    llm_model: str = Field(default="")

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _check_provider(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            if not v:
                return None
            if v not in {"groq", "openai", "gemini"}:
                raise ValueError(
                    "LLM_PROVIDER must be groq, openai or gemini"
                )
        return v
    llm_temperature: float = Field(default=0.4)
    # Caps every per-schema output budget. Leave unset to use the defaults in
    # `app/agent/llm.py`; set it when your key's tokens-per-minute limit is
    # tighter than those assume, which shows up as a 413 rather than a 429.
    llm_max_tokens: Optional[int] = Field(default=None, ge=256, le=32000)
    # Reasoning models spend output tokens thinking before they answer. On a
    # tight rate limit that can consume the whole budget, leaving nothing for
    # the structured result. Sent to the provider only when set.
    llm_reasoning_effort: Optional[str] = Field(default=None)
    # How structured output is requested. `function_calling` is the default and
    # works on most providers; `json_mode` asks for raw JSON instead, which is
    # the escape hatch when a model keeps failing to emit a tool call.
    llm_structured_method: str = Field(default="function_calling")

    # --- Planning ---
    # How many days a generated plan covers. Four rather than seven because a
    # week costs roughly 7500 tokens of an 8000-per-minute free tier, which
    # leaves no room for the retry the agent is designed around. Raise it if
    # your provider's limit allows. Nothing else in the system assumes seven.
    plan_duration_days: int = Field(default=4, ge=1, le=14)

    @field_validator("llm_structured_method", mode="before")
    @classmethod
    def _check_method(cls, v):
        if isinstance(v, str):
            v = v.strip().lower() or "function_calling"
            if v not in {"function_calling", "json_mode"}:
                raise ValueError(
                    "LLM_STRUCTURED_METHOD must be function_calling or json_mode"
                )
        return v

    @field_validator("llm_reasoning_effort", mode="before")
    @classmethod
    def _normalise_effort(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            if not v:
                return None
            if v not in {"low", "medium", "high"}:
                raise ValueError(
                    "LLM_REASONING_EFFORT must be low, medium or high"
                )
        return v

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
    # source*, before any validator gets a look, so the comma-separated form
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
        what the old list-typed field used to require, so both keep working.
        """
        if not isinstance(v, str):
            return [normalise_origin(item) for item in v] if v else v

        text = v.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass  # fall through and treat it as comma-separated
            else:
                return [normalise_origin(item) for item in parsed]

        return [
            normalise_origin(origin) for origin in text.split(",") if origin.strip()
        ]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance. Read the environment exactly once."""
    return Settings()


settings = get_settings()
