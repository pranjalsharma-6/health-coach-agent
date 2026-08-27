"""LLM provider abstraction.

The rest of the agent asks for `get_llm()` and never names a vendor. Swapping
Groq for OpenAI (or anything else LangChain speaks) is a config change, not a
refactor.
"""

from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_llm: Optional[BaseChatModel] = None


class LLMUnavailableError(RuntimeError):
    """Raised when no provider is configured. Distinct from a call failure."""


@dataclass(frozen=True)
class LLMFailure:
    """What went wrong on a model call, and whether trying again could help."""

    message: str
    retryable: bool


def describe_llm_failure(exc: Exception) -> LLMFailure:
    """Turn a provider exception into something a human can act on.

    Two things depend on getting this right. The message is shown in the run
    timeline, where `NotFoundError` alone tells the user nothing. And
    `retryable` decides whether the agent tries again: a 404 for a decommissioned
    model will fail identically three times, so retrying spends the user's time
    to arrive at the same place.

    Classification is by HTTP status rather than exception class, because Groq
    and OpenAI raise their own separate hierarchies for the same conditions and
    this module deliberately does not import either vendor's SDK.
    """
    status = getattr(exc, "status_code", None)
    detail = str(exc).strip()

    if status == 404:
        return LLMFailure(
            f"The model '{settings.llm_model}' is not available on your API key. "
            "Providers retire models on a rolling basis. Run "
            "`python -m app.tools.check_llm` to list the models your key can "
            "use, then set LLM_MODEL in backend/.env to one of them.",
            retryable=False,
        )

    if status in (401, 403):
        return LLMFailure(
            "The API key was rejected. Check GROQ_API_KEY in backend/.env — "
            "it should start with 'gsk_' and have no quotes or trailing spaces.",
            retryable=False,
        )

    if status == 429:
        return LLMFailure(
            "Rate limited by the provider. The free tier has a per-minute cap; "
            "waiting a minute usually clears it.",
            retryable=True,
        )

    if status == 400:
        return LLMFailure(
            f"The provider rejected the request: {detail}",
            retryable=False,
        )

    if status is not None and 500 <= status < 600:
        return LLMFailure(
            f"The provider had a server error ({status}). This is usually "
            "temporary.",
            retryable=True,
        )

    # No status: a timeout, a dropped connection, or output that would not parse
    # into the schema. All three are worth another attempt.
    return LLMFailure(
        f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__,
        retryable=True,
    )


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
