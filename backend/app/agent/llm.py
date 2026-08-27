"""LLM provider abstraction.

The rest of the agent asks for `get_llm()` and never names a vendor. Swapping
Groq for OpenAI (or anything else LangChain speaks) is a config change, not a
refactor.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Output budgets, per structured-output schema.
#
# `max_tokens` is not a cap you can leave generous "just in case": providers
# count the space you *reserve* against your rate limit, so an 8000-token
# reservation exhausted an 8000 TPM free tier before the prompt was counted at
# all, and every call came back 413. These are sized to what each schema
# actually produces — a training week is seven activities, not a novel.
MAX_OUTPUT_TOKENS: Dict[str, int] = {
    # The meal budget also has to cover a reasoning model's thinking tokens,
    # which are spent before the answer begins. Sized to the largest value the
    # fan-out can afford against an 8000 TPM tier, not to the answer alone.
    "MealPlanDraft": 4400,      # 7 days x ~4 meals, each with macros
    "TrainingPlanDraft": 1600,  # 7 days x a session of named exercises
    "PlanCritique": 700,        # a verdict and a short list of issues
    "Recipe": 1200,             # ingredients, steps, tips for one meal
}
DEFAULT_MAX_OUTPUT_TOKENS = 2000

# Cached per budget: the budget is a constructor argument, so one client per
# distinct value rather than one client overall.
_llm_by_budget: Dict[int, BaseChatModel] = {}


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

    if status == 413:
        return LLMFailure(
            "The request was larger than your provider plan allows. Providers "
            "count the output tokens you reserve against your per-minute "
            "limit, so lowering LLM_MAX_TOKENS in backend/.env (try 1500) "
            "usually fixes this. The error text names your limit.",
            retryable=False,
        )

    if status == 429:
        return LLMFailure(
            "Rate limited by the provider. The free tier has a per-minute cap; "
            "waiting a minute usually clears it.",
            retryable=True,
        )

    if status == 400:
        # Neither `tool_use_failed` nor `json_validate_failed` is a malformed
        # request. Both mean the model was asked for structured output and did
        # not manage to produce it *this time* — a missed tool call, or JSON
        # that drifted and stopped closing its braces partway through a long
        # array. That is precisely what the retry budget exists for, and
        # lumping them in with genuine 400s meant one attempt and no second
        # chance. An empty `failed_generation` points at truncation instead:
        # reasoning models spend `max_tokens` thinking before they emit
        # anything, so a budget that fits the answer can still leave none for
        # it.
        if "tool_use_failed" in detail or "json_validate_failed" in detail:
            return LLMFailure(
                "The model did not return the structured output the plan needs. "
                "If this repeats, the output budget is likely being consumed "
                "before the answer starts — set LLM_REASONING_EFFORT=low in "
                "backend/.env, or raise LLM_MAX_TOKENS if your rate limit "
                "allows.",
                retryable=True,
            )

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


def budget_for(schema: Any) -> int:
    """How many output tokens this schema is allowed to use.

    `LLM_MAX_TOKENS` caps every budget when set, for keys on a tighter rate
    limit than the defaults assume.
    """
    budget = MAX_OUTPUT_TOKENS.get(
        getattr(schema, "__name__", ""), DEFAULT_MAX_OUTPUT_TOKENS
    )
    ceiling = settings.llm_max_tokens
    return min(budget, ceiling) if ceiling else budget


def get_llm(max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> BaseChatModel:
    """Return a cached chat model, preferring Groq.

    Groq serves large open models at very low latency on a free tier, which
    matters for an agent the user watches work in real time.
    """
    cached = _llm_by_budget.get(max_tokens)
    if cached is not None:
        return cached

    if settings.groq_api_key:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
            max_tokens=max_tokens,
            timeout=90,
            max_retries=2,
            # Only sent when configured. Reasoning models spend part of the
            # output budget thinking before they answer; turning that down
            # leaves more of it for the structured result. Passed through
            # model_kwargs because it is a provider parameter, not a LangChain
            # one, and omitted entirely when unset so providers that reject the
            # field are unaffected.
            model_kwargs=(
                {"reasoning_effort": settings.llm_reasoning_effort}
                if settings.llm_reasoning_effort
                else {}
            ),
        )
        logger.info(
            "LLM provider: Groq (%s, max_tokens=%s)", settings.llm_model, max_tokens
        )
        _llm_by_budget[max_tokens] = llm
        return llm

    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            max_tokens=max_tokens,
            timeout=90,
            max_retries=2,
        )
        logger.info("LLM provider: OpenAI (gpt-4o-mini, max_tokens=%s)", max_tokens)
        _llm_by_budget[max_tokens] = llm
        return llm

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
    llm = get_llm(budget_for(schema))

    if settings.llm_structured_method == "json_mode":
        return _JsonModeAdapter(
            llm.with_structured_output(schema, method="json_mode"), schema
        )

    return llm.with_structured_output(schema)


class _JsonModeAdapter:
    """Appends the schema to the prompt on the json_mode path.

    Under `function_calling` the provider is handed the schema and constrains
    the output to it. Under `json_mode` it only promises valid JSON, so the
    shape has to be in the prompt — otherwise the model returns well-formed
    JSON of entirely the wrong shape and the parse fails downstream, which
    looks like a model problem rather than a missing instruction.

    Wrapping it here rather than at each call site means no caller has to
    remember, and switching methods stays a config change.
    """

    def __init__(self, runnable: Any, schema: Any):
        self._runnable = runnable
        self._instruction = build_json_mode_instruction(schema)

    async def ainvoke(self, messages: Any) -> Any:
        return await self._runnable.ainvoke(self._with_schema(messages))

    def _with_schema(self, messages: Any) -> Any:
        if isinstance(messages, str):
            return messages + self._instruction

        if not isinstance(messages, list) or not messages:
            return messages

        # Append to the final message rather than adding another: some
        # providers require the last message to be the user turn.
        last = messages[-1]
        content = getattr(last, "content", None)
        if not isinstance(content, str):
            return messages

        amended = last.model_copy(update={"content": content + self._instruction})
        return [*messages[:-1], amended]


def build_json_mode_instruction(schema: Any) -> str:
    """The schema, as prompt text, for the json_mode path.

    Under `function_calling` the provider constrains the output to the schema.
    Under `json_mode` it only guarantees valid JSON, so the shape has to be
    stated in the prompt or the model has no way to know it.
    """
    import json

    return (
        "\n\nReturn a single JSON object and nothing else — no prose, no "
        "markdown fences. It must match this JSON Schema exactly:\n\n"
        + json.dumps(schema.model_json_schema(), indent=2)
    )


def is_configured() -> bool:
    return bool(settings.groq_api_key or settings.openai_api_key)


def reset_cache() -> None:
    """Drop the cached clients. Used by tests."""
    _llm_by_budget.clear()
