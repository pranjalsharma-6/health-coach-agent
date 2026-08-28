"""LLM provider abstraction.

The rest of the agent asks for `get_llm()` and never names a vendor. Swapping
Groq for OpenAI (or anything else LangChain speaks) is a config change, not a
refactor.
"""

import asyncio
import re

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
# Output budgets that scale with the work.
#
# A flat number is wrong in both directions: sized for a seven-day plan it
# over-reserves for a four-day one, and providers count reserved output against
# the per-minute limit whether it is used or not. The trainer was reserving
# 1400 tokens to write four days of exercises.
#
# (per_day, overhead) — overhead covers the reasoning field and JSON scaffolding.
PER_DAY_OUTPUT_TOKENS: Dict[str, tuple] = {
    "MealPlanDraft": (350, 500),      # ~4 meals a day, each with macros
    "TrainingPlanDraft": (170, 400),  # one session a day, 3-4 named exercises
}

MAX_OUTPUT_TOKENS: Dict[str, int] = {
    # Fallbacks for schemas whose size does not depend on the plan length.
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
    status = http_status(exc)
    detail = str(exc).strip()

    if status == 404:
        # The *resolved* model, not the raw setting: LLM_MODEL is blank when
        # the provider's default is in use, and "The model '' is not available"
        # helps nobody.
        provider = resolve_provider()
        named = (
            (resolve_model(provider) if provider else settings.llm_model)
            or "the configured model"
        )

        replacement = suggested_replacement(detail)
        if replacement:
            return LLMFailure(
                f"The model '{named}' is no longer available. The provider "
                f"suggests '{replacement}' — set LLM_MODEL={replacement} in "
                "backend/.env and restart.",
                retryable=False,
            )

        return LLMFailure(
            f"The model '{named}' is not available on your API key. "
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
        # A per-minute limit and a per-day one are the same status code and a
        # completely different problem. Waiting works for the first and is
        # useless for the second — telling a user to "wait a minute" when they
        # have spent their day's allowance sends them in circles.
        # What decides the response is how long the wait is, not which
        # quota was hit. Google reports a *daily* request quota and then says
        # "retry in 31s", because the window rolls — refusing to wait there
        # would throw away a run that was half a minute from working.
        hinted = parse_retry_after_seconds(detail, cap=None)
        waitable = hinted is not None and hinted <= MAX_RATE_LIMIT_WAIT_SECONDS

        if waitable:
            return LLMFailure(
                f"Rate limited by the provider{describe_wait(detail)}. Kaya "
                "waits and retries on its own.",
                retryable=True,
            )

        if is_daily_limit(detail):
            return LLMFailure(
                "You have used your provider's allowance for the day"
                f"{describe_wait(detail)}. Waiting will not help much: use a "
                "smaller plan (PLAN_DURATION_DAYS in backend/.env), switch "
                "provider with LLM_PROVIDER, or raise the limit on your "
                "account.",
                retryable=False,
            )

        return LLMFailure(
            "Rate limited by the provider, with no usable estimate of how long "
            "for. Try again shortly.",
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
    name = getattr(schema, "__name__", "")

    scaled = PER_DAY_OUTPUT_TOKENS.get(name)
    if scaled is not None:
        per_day, overhead = scaled
        budget = per_day * _days_per_call(name) + overhead
    else:
        budget = MAX_OUTPUT_TOKENS.get(name, DEFAULT_MAX_OUTPUT_TOKENS)
    ceiling = settings.llm_max_tokens
    return min(budget, ceiling) if ceiling else budget


def _days_per_call(schema_name: str) -> int:
    """How many days one call to this specialist has to cover.

    The nutritionist drafts in chunks, so its call covers a chunk; the trainer
    writes the whole plan in one go. Imported here rather than at module level
    because the graph imports this module.
    """
    from app.agent.graph import MEAL_CHUNK_DAYS, PLAN_DURATION_DAYS

    if schema_name == "MealPlanDraft":
        return min(MEAL_CHUNK_DAYS, PLAN_DURATION_DAYS)
    return PLAN_DURATION_DAYS


# Model names do not travel between providers, so each carries its own
# default. `LLM_MODEL` overrides whichever is selected.
# These go stale. Providers retire models on their own schedule, and a default
# written down once is a default that is wrong later — `gemini-2.5-flash` was
# correct when it was written and refused for new keys by the time it shipped.
# `check_llm` lists what a key can actually reach, and the 404 below repeats
# whatever replacement the provider names.
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.6-flash",
    "openai": "gpt-4o-mini",
}


def resolve_provider() -> Optional[str]:
    """Which provider to use: the configured one, else whichever key exists.

    Explicit beats implicit — but with one key set, asking someone to also name
    the provider is a second chance to get it wrong.
    """
    if settings.llm_provider:
        return settings.llm_provider

    for name, key in (
        ("groq", settings.groq_api_key),
        ("gemini", settings.gemini_api_key),
        ("openai", settings.openai_api_key),
    ):
        if key:
            return name
    return None


def resolve_model(provider: str) -> str:
    return settings.llm_model or DEFAULT_MODELS.get(provider, "")


def api_key_for(provider: str) -> str:
    return {
        "groq": settings.groq_api_key,
        "gemini": settings.gemini_api_key,
        "openai": settings.openai_api_key,
    }.get(provider, "")


def get_llm(max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> BaseChatModel:
    """Return a cached chat model for the configured provider."""
    cached = _llm_by_budget.get(max_tokens)
    if cached is not None:
        return cached

    provider = resolve_provider()
    key = api_key_for(provider) if provider else ""

    if not provider or not key:
        raise LLMUnavailableError(
            "No LLM provider configured. Set one of GROQ_API_KEY, "
            "GEMINI_API_KEY (free at aistudio.google.com) or OPENAI_API_KEY "
            "in backend/.env"
        )

    model = resolve_model(provider)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=settings.llm_temperature,
            google_api_key=key,
            # Gemini spells the output cap differently. Same meaning, same
            # reservation semantics.
            max_output_tokens=max_tokens,
            timeout=90,
            max_retries=2,
        )
        logger.info("LLM provider: Gemini (%s, max_tokens=%s)", model, max_tokens)
        _llm_by_budget[max_tokens] = llm
        return llm

    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=model,
            temperature=settings.llm_temperature,
            api_key=key,
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
        logger.info("LLM provider: Groq (%s, max_tokens=%s)", model, max_tokens)
        _llm_by_budget[max_tokens] = llm
        return llm

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=settings.llm_temperature,
        api_key=key,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=2,
    )
    logger.info("LLM provider: OpenAI (%s, max_tokens=%s)", model, max_tokens)
    _llm_by_budget[max_tokens] = llm
    return llm


def get_structured_llm(schema: Any) -> Any:
    """Return an LLM bound to a Pydantic output schema.

    Structured output is the first line of defence against malformed plans; the
    validator in `validators.py` is the second, because schema-valid output can
    still be nutritionally wrong or violate the user's diet.
    """
    llm = get_llm(budget_for(schema))

    # `method=` is an OpenAI-style option. Gemini's integration accepts
    # **kwargs and quietly drops it, so asking for json_mode there produced the
    # worst of both: its own function-calling path *plus* a JSON Schema dumped
    # into the prompt by the adapter. The model got two contradictory
    # instructions and returned something the parser could not coerce.
    if settings.llm_structured_method == "json_mode" and provider_supports_json_mode():
        return _RateLimitRetrying(
            _NeverNone(
                _JsonModeAdapter(
                    llm.with_structured_output(schema, method="json_mode"), schema
                ),
                schema,
            )
        )

    return _RateLimitRetrying(_NeverNone(llm.with_structured_output(schema), schema))


def provider_supports_json_mode() -> bool:
    """Which providers understand `with_structured_output(method=...)`.

    Gemini does not. Silently ignoring the setting is better than failing, but
    only because the alternative path works — this is not a preference.
    """
    return resolve_provider() in {"groq", "openai"}


class _NeverNone:
    """Turn a silent `None` into an error that names itself.

    LangChain's structured-output parsers return None when they cannot coerce a
    response — no exception, no message. The None then travels until something
    reads an attribute off it, and the traceback points at `draft.days` in the
    graph rather than at the model call that produced nothing. Failing here
    keeps the diagnosis next to the cause, and makes it retryable like any
    other generation failure.
    """

    def __init__(self, runnable: Any, schema: Any):
        self._runnable = runnable
        self._schema_name = getattr(schema, "__name__", "the expected schema")

    async def ainvoke(self, messages: Any) -> Any:
        result = await self._runnable.ainvoke(messages)
        if result is None:
            raise ValueError(
                f"The model returned nothing usable for {self._schema_name}. "
                "Its reply could not be read as the required structure — often "
                "a response that was empty, truncated, or blocked."
            )
        return result


# How long to wait for a rate-limit window, and how many times.
#
# The generation retry budget and a per-minute token limit interact badly: a
# 7-day plan costs most of an 8000 TPM window on its first attempt, so an
# immediate second attempt is guaranteed to be refused. Waiting for the window
# to roll is the difference between three attempts and one.
MAX_RATE_LIMIT_WAITS = 2
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 35.0
MAX_RATE_LIMIT_WAIT_SECONDS = 90.0


def http_status(exc: Exception) -> Optional[int]:
    """The HTTP status behind a provider exception, however it is spelled.

    Groq and OpenAI expose `status_code`. Google's `api_core` exceptions carry
    `code` instead, so a layer that only knew `status_code` was blind to every
    Gemini rate limit: no wait, instant retries, and a generic message. Falling
    back to the text catches wrappers that expose neither.
    """
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value

    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def suggested_replacement(message: str) -> Optional[str]:
    """The model a provider names as the successor, when it names one.

    Google's 404 is unusually helpful: "This model models/gemini-2.5-flash is
    no longer available to new users. Please update your code to use
    models/gemini-3.6-flash". Repeating that back turns a dead end into a
    one-line fix, and the provider is a better source for its own model names
    than anything written into this file.
    """
    match = re.search(
        r"use\s+(?:models/)?([A-Za-z0-9][A-Za-z0-9._\-]{2,60})", message
    )
    if not match:
        return None

    candidate = match.group(1).rstrip(".")
    # "use the Interactions API" and similar prose are not model names.
    return candidate if any(c.isdigit() for c in candidate) else None


def is_daily_limit(message: str) -> bool:
    """Is this a per-day allowance rather than a per-minute one?

    Groq names it in the error text: "on tokens per day (TPD)". The status code
    is 429 either way, so without this the two are indistinguishable — and they
    call for opposite responses.
    """
    lowered = message.lower()
    # Groq says "on tokens per day (TPD)". Google names the quota id, e.g.
    # "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
    return "per day" in lowered or "tpd" in lowered or "perday" in lowered


def describe_wait(message: str) -> str:
    """", and the provider suggests waiting about 9 minutes" — or nothing.

    Read from the provider's own text rather than estimated, and phrased as a
    clause so it drops cleanly into a sentence when absent.
    """
    seconds = parse_retry_after_seconds(message, cap=None)
    if seconds is None:
        return ""
    if seconds < 90:
        return f", and it suggests waiting about {round(seconds)} seconds"
    return f", and it suggests waiting about {round(seconds / 60)} minutes"


def parse_retry_after_seconds(
    message: str, cap: Optional[float] = MAX_RATE_LIMIT_WAIT_SECONDS
) -> Optional[float]:
    """How long the provider asked us to wait, if it said.

    Groq phrases it inside the error text — "Please try again in 1m13.5s" —
    rather than only in a header, and the header is not reachable through the
    exception LangChain surfaces. Honouring the provider's own number beats
    guessing at one.
    """
    # "Please try again in 1m13.5s" (Groq) and "Please retry in 31.24s"
    # (Google) are the same sentence with a different verb.
    match = re.search(
        r"(?:try again|retry) in\s+(?:(\d+)m)?\s*([\d.]+)s", message, re.IGNORECASE
    )
    if not match:
        return None

    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2))
    total = minutes * 60 + seconds
    return min(total, cap) if cap is not None else total


class _RateLimitRetrying:
    """Waits out a rate limit rather than spending a generation attempt on it.

    Being told "you have used your tokens for this minute" is not a failure of
    the plan, and re-drafting immediately only earns the same refusal. This
    waits for the window and repeats the identical call, so the agent's three
    attempts stay available for problems that are actually about the food.

    `sleep` is injected so tests do not spend real time.
    """

    def __init__(self, runnable: Any, sleep=asyncio.sleep):
        self._runnable = runnable
        self._sleep = sleep

    async def ainvoke(self, messages: Any) -> Any:
        for attempt in range(MAX_RATE_LIMIT_WAITS + 1):
            try:
                return await self._runnable.ainvoke(messages)
            except Exception as exc:
                if attempt == MAX_RATE_LIMIT_WAITS:
                    raise
                if http_status(exc) != 429:
                    raise

                hinted = parse_retry_after_seconds(str(exc), cap=None)
                if hinted is not None and hinted > MAX_RATE_LIMIT_WAIT_SECONDS:
                    # Blocking a web request for minutes is a hang, not a
                    # retry. Fail now and let the message explain.
                    raise

                wait = (
                    parse_retry_after_seconds(str(exc))
                    or DEFAULT_RATE_LIMIT_WAIT_SECONDS
                )
                logger.warning(
                    "Rate limited; waiting %.1fs before retrying (%s/%s)",
                    wait,
                    attempt + 1,
                    MAX_RATE_LIMIT_WAITS,
                )
                await self._sleep(wait)

        raise RuntimeError("unreachable")  # pragma: no cover


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
    provider = resolve_provider()
    return bool(provider and api_key_for(provider))


def reset_cache() -> None:
    """Drop the cached clients. Used by tests."""
    _llm_by_budget.clear()
