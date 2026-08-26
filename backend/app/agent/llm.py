"""LLM provider abstraction.

The rest of the agent asks for `get_llm()` and never names a vendor. Swapping
Groq for OpenAI (or anything else LangChain speaks) is a config change, not a
refactor.
"""

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_llm: Optional[BaseChatModel] = None


class LLMUnavailableError(RuntimeError):
    """Raised when no provider is configured. Distinct from a call failure."""


def get_llm() -> BaseChatModel:
    """Return a cached chat model, preferring Groq.

    Groq runs Llama 3.3 70B at very low latency on a free tier, which matters
    for an agent the user watches work in real time.
    """
    global _llm
    if _llm is not None:
        return _llm

    if settings.groq_api_key:
        from langchain_groq import ChatGroq

        _llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
            max_tokens=8000,
            timeout=90,
            max_retries=2,
        )
        logger.info("LLM provider: Groq (%s)", settings.llm_model)
        return _llm

    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            timeout=90,
            max_retries=2,
        )
        logger.info("LLM provider: OpenAI (gpt-4o-mini)")
        return _llm

    raise LLMUnavailableError(
        "No LLM provider configured. Set GROQ_API_KEY (free at console.groq.com) "
        "or OPENAI_API_KEY in backend/.env"
    )


def get_structured_llm(schema: Any) -> Any:
    """Return an LLM bound to a Pydantic output schema.

    Structured output is the first line of defence against malformed plans; the
    validator in `validators.py` is the second, because schema-valid output can
    still be nutritionally wrong or violate the user's diet.
    """
    return get_llm().with_structured_output(schema)


def is_configured() -> bool:
    return bool(settings.groq_api_key or settings.openai_api_key)


def reset_cache() -> None:
    """Drop the cached client. Used by tests."""
    global _llm
    _llm = None
