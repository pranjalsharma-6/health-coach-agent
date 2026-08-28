"""How the agent behaves when the model provider says no.

The run timeline used to show `Meal drafting failed: NotFoundError` — the
exception's class name with its message discarded — and then retry three times.
A 404 for a retired model fails identically every attempt, so those retries
bought nothing but the user's time, and the one fact that would have explained
it (which model, and what to do) was never displayed.
"""

from datetime import date

import pytest

from app.agent import graph
from app.agent.llm import describe_llm_failure
from app.core.config import settings
from app.tools.check_llm import (
    looks_like_a_chat_model,
    parameter_count_b,
    rank_for_structured_output,
)
from app.models.enums import DietType
from tests.factories import make_log, make_meal_draft, make_profile, make_targets

TODAY = date(2026, 3, 15)
TARGETS = make_targets()


class ProviderError(Exception):
    """Stands in for groq.NotFoundError / openai.NotFoundError etc.

    Classification is by `status_code`, deliberately not by class, so this
    stays honest without importing a vendor SDK.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class TestClassification:
    def test_a_retired_model_is_not_retryable(self):
        failure = describe_llm_failure(ProviderError("model_not_found", 404))
        assert failure.retryable is False
        assert "LLM_MODEL" in failure.message
        assert "check_llm" in failure.message

    def test_a_rejected_key_is_not_retryable(self):
        failure = describe_llm_failure(ProviderError("invalid api key", 401))
        assert failure.retryable is False
        assert "GROQ_API_KEY" in failure.message

    def test_rate_limiting_is_retryable(self):
        assert describe_llm_failure(ProviderError("slow down", 429)).retryable

    def test_a_provider_outage_is_retryable(self):
        assert describe_llm_failure(ProviderError("upstream", 503)).retryable

    def test_a_parse_failure_keeps_the_message(self):
        """No status code — a timeout or unparseable output. Worth retrying,
        and the message is the only clue about which."""
        failure = describe_llm_failure(ValueError("expected an object"))
        assert failure.retryable is True
        assert "expected an object" in failure.message


@pytest.fixture
def wired(monkeypatch):
    class FakePlanRepo:
        @staticmethod
        async def get_active(_user_id):
            return None

        @staticmethod
        async def save_new_version(plan):
            plan.id, plan.version = "plan-1", 1
            return plan

    class FakeLogRepo:
        @staticmethod
        async def get_or_create(_user_id, log_date):
            return make_log(log_date, [])

        @staticmethod
        async def get_recent(_user_id, days=7):
            return []

    class FakeProfileRepo:
        @staticmethod
        async def get(_user_id):
            return make_profile(diet_type=DietType.VEGETARIAN)

    class FakeEventRepo:
        @staticmethod
        async def record(event):
            return event

    monkeypatch.setattr(graph, "PlanRepository", FakePlanRepo)
    monkeypatch.setattr(graph, "LogRepository", FakeLogRepo)
    monkeypatch.setattr(graph, "ProfileRepository", FakeProfileRepo)
    monkeypatch.setattr(graph, "AgentEventRepository", FakeEventRepo)


def always_raising(exc: Exception, counter: list):
    def factory(_schema):
        class Stub:
            async def ainvoke(self, _messages):
                counter.append(1)
                raise exc

        return Stub()

    return factory


class TestNonRetryableFailures:
    async def test_both_specialists_failing_at_once_does_not_crash(
        self, wired, monkeypatch
    ):
        """The regression this file exists for.

        `plan_meals` and `plan_training` share a superstep. A retired model
        breaks both, so both write to `error` concurrently — which a plain
        LastValue channel rejects with LangGraph's `InvalidUpdateError`,
        replacing the real diagnosis with a framework complaint about
        concurrent writes. It also fired whenever no API key was configured.
        """
        calls: list = []
        monkeypatch.setattr(
            graph,
            "get_structured_llm",
            always_raising(ProviderError("model_not_found", 404), calls),
        )

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is None
        assert "not available on your API key" in final["error"]

    async def test_it_gives_up_after_one_attempt(self, wired, monkeypatch):
        """Three identical 404s tell the user nothing three times."""
        calls: list = []
        monkeypatch.setattr(
            graph,
            "get_structured_llm",
            always_raising(ProviderError("model_not_found", 404), calls),
        )

        await graph.run_agent("u1", today=TODAY)

        expected = graph.meal_chunk_count() + 1  # meal chunks + the trainer
        assert len(calls) == expected, (
            f"expected one attempt from each specialist, got {len(calls)} calls "
            "— a config error is being retried"
        )

    async def test_the_reason_reaches_the_run_timeline(self, wired, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            graph,
            "get_structured_llm",
            always_raising(ProviderError("model_not_found", 404), calls),
        )

        final = await graph.run_agent("u1", today=TODAY)
        messages = " ".join(s["message"] for s in final["steps"])

        assert "llama" in messages.lower(), "the model name should be named"
        assert "NotFoundError" not in messages, (
            "the exception class name is not a diagnosis"
        )


class TestRetryableFailuresStillRetry:
    async def test_an_unparseable_response_is_retried(self, wired, monkeypatch):
        """The retry budget still exists — it just isn't spent on config."""
        calls: list = []
        monkeypatch.setattr(
            graph, "get_structured_llm", always_raising(ValueError("bad json"), calls)
        )

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is None
        meal_attempts = sum(
            1 for s in final["steps"] if s["node"] == "plan_meals" and s["status"] == "error"
        )
        assert meal_attempts == graph.MAX_GENERATION_ATTEMPTS

    async def test_a_transient_failure_can_recover(self, wired, monkeypatch):
        """First call fails, second succeeds — the plan should still ship."""
        from app.models.plan import MealPlanDraft, PlanCritique, TrainingPlanDraft
        from tests.factories import (
            make_critique,
            make_training_draft,
            scope_to_requested_days,
        )

        state = {"meal_calls": 0}

        def factory(schema):
            class Stub:
                async def ainvoke(self, messages):
                    if schema is TrainingPlanDraft:
                        return make_training_draft()
                    if schema is PlanCritique:
                        return make_critique()
                    state["meal_calls"] += 1
                    if state["meal_calls"] == 1:
                        raise ProviderError("upstream hiccup", 503)
                    return scope_to_requested_days(
                        make_meal_draft(TARGETS), messages
                    )

            return Stub()

        monkeypatch.setattr(graph, "get_structured_llm", factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is not None, "a transient error should recover"
        # A failure in one chunk re-runs the whole round, so two full sets
        # of chunk calls: the first (one of which raised) and the retry.
        assert state["meal_calls"] == 2 * graph.meal_chunk_count()


class TestModelListFilter:
    @pytest.mark.parametrize(
        "name",
        ["llama-3.3-70b-versatile", "llama-4-scout-17b", "gpt-4o", "mixtral-8x7b"],
    )
    def test_keeps_chat_models(self, name):
        assert looks_like_a_chat_model(name)

    @pytest.mark.parametrize(
        "name",
        [
            "whisper-large-v3",
            "playai-tts",
            "text-embedding-3-small",
            "meta-llama/llama-guard-4-12b",
        ],
    )
    def test_drops_the_ones_that_cannot_hold_a_conversation(self, name):
        """Speech, embedding and moderation models pad the list with noise."""
        assert not looks_like_a_chat_model(name)


# The exact list returned by a real Groq key, August 2026. Kept verbatim rather
# than idealised, because both bugs below only appeared against real data.
REAL_GROQ_LIST = [
    "allam-2-7b",
    "canopylabs/orpheus-arabic-saudi",
    "canopylabs/orpheus-v1-english",
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
]


class TestRecommendation:
    """The tool recommended a 7B Arabic model over a 120B general one.

    Not because of any judgement about capability — it took the first entry of
    an alphabetically sorted list, and "allam" sorts before "openai". Following
    that advice produces three failed generation attempts and an error message
    that blames the plan.
    """

    @staticmethod
    def _candidates():
        return rank_for_structured_output(
            [m for m in REAL_GROQ_LIST if looks_like_a_chat_model(m)]
        )

    def test_recommends_the_largest_model(self):
        assert self._candidates()[0] == "openai/gpt-oss-120b"

    def test_does_not_recommend_a_small_model(self):
        assert self._candidates()[0] != "allam-2-7b"

    def test_speech_models_are_excluded(self):
        """`orpheus` carries none of the usual giveaways — no "tts", no
        "whisper" — so it was offered as a candidate for meal planning."""
        candidates = self._candidates()
        assert not [c for c in candidates if "orpheus" in c]

    def test_unsized_models_sort_last(self):
        """An unlabelled name is not evidence of capability. Sorting it first
        would reintroduce the original bug by a different route."""
        ranked = self._candidates()
        assert ranked.index("groq/compound") > ranked.index("allam-2-7b")

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("openai/gpt-oss-120b", 120.0),
            ("qwen/qwen3.8-27b", 27.0),   # the 3.8 is a version, not a size
            ("allam-2-7b", 7.0),
            ("groq/compound", 0.0),
            ("mixtral-8x7b", 7.0),
        ],
    )
    def test_parameter_count_parsing(self, name, expected):
        assert parameter_count_b(name) == expected


class TestOutputBudgets:
    """`max_tokens` is a reservation, not a cap you can leave generous.

    Providers count the output space you reserve against your per-minute token
    limit. An 8000-token reservation against an 8000 TPM free tier meant every
    call was over the limit before the prompt was counted, and the run failed
    with 413 six times in a row. The budgets below are sized to what each
    schema actually produces.
    """

    # Groq's free tier at the time of writing. The point of the number is that
    # the fan-out has to fit inside *some* stated limit, not this exact one.
    FREE_TIER_TPM = 8000

    # Measured from the real builders against a profile with allergies,
    # dislikes and three cuisines — the largest prompt a real user produces.
    # The trainer's grew when it started carrying the exercise list.
    NUTRITIONIST_PROMPT = 1000
    TRAINER_PROMPT = 700

    def test_the_concurrent_fan_out_fits_the_free_tier(self):
        """Everything in one superstep shares a rate-limit window.

        Both specialists run together, and the nutritionist is itself several
        concurrent chunk calls — each reserving the meal budget and each
        carrying a full copy of the prompt. Counting one meal call here is what
        let two chunks reserve 8800 tokens against an 8000 limit.
        """
        from app.agent import graph
        from app.agent.llm import budget_for
        from app.models.plan import MealPlanDraft, TrainingPlanDraft

        chunks = graph.meal_chunk_count()
        total = (
            chunks * (self.NUTRITIONIST_PROMPT + budget_for(MealPlanDraft))
            + self.TRAINER_PROMPT
            + budget_for(TrainingPlanDraft)
        )
        assert total <= self.FREE_TIER_TPM, (
            f"the fan-out reserves {total} tokens against a {self.FREE_TIER_TPM} "
            "limit — every run will fail with 413"
        )

    def test_no_single_budget_exceeds_the_limit_alone(self):
        from app.agent.llm import MAX_OUTPUT_TOKENS

        for name, budget in MAX_OUTPUT_TOKENS.items():
            assert budget < self.FREE_TIER_TPM, f"{name} alone exceeds the limit"

    def test_the_trainer_gets_less_than_the_nutritionist(self):
        """Seven activities against seven days of meals with macros. Giving
        them the same reservation is what wasted the budget."""
        from app.agent.llm import budget_for
        from app.models.plan import MealPlanDraft, TrainingPlanDraft

        assert budget_for(TrainingPlanDraft) < budget_for(MealPlanDraft)

    def test_an_unknown_schema_gets_a_conservative_default(self):
        from app.agent.llm import DEFAULT_MAX_OUTPUT_TOKENS, budget_for

        class SomethingNew:
            pass

        assert budget_for(SomethingNew) == DEFAULT_MAX_OUTPUT_TOKENS

    def test_llm_max_tokens_caps_every_budget(self, monkeypatch):
        """The escape hatch for a key on a tighter limit than the defaults
        assume."""
        from app.agent.llm import budget_for
        from app.core.config import settings
        from app.models.plan import MealPlanDraft, PlanCritique

        monkeypatch.setattr(settings, "llm_max_tokens", 900)

        assert budget_for(MealPlanDraft) == 900
        # A budget already below the ceiling is left alone rather than raised.
        assert budget_for(PlanCritique) == 700

    def test_the_client_is_cached_per_budget(self, monkeypatch):
        """One client per distinct budget, not one per call — the budget is a
        constructor argument, so a single cached client cannot serve both."""
        import app.agent.llm as llm_module

        built = []

        class FakeChat:
            def __init__(self, **kwargs):
                built.append(kwargs["max_tokens"])

        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "openai_api_key", "")
        llm_module.reset_cache()

        import sys
        import types

        fake = types.ModuleType("langchain_groq")
        fake.ChatGroq = FakeChat
        monkeypatch.setitem(sys.modules, "langchain_groq", fake)

        llm_module.get_llm(3500)
        llm_module.get_llm(3500)   # cached
        llm_module.get_llm(1000)   # different budget, new client

        assert built == [3500, 1000]
        llm_module.reset_cache()


class TestRequestTooLarge:
    def test_413_is_not_retried(self):
        """It is deterministic inside the rate-limit window: the same request
        fails identically, so three attempts produce three failures and no
        information. It was retried, which is why the timeline showed six."""
        failure = describe_llm_failure(ProviderError("too large", 413))
        assert failure.retryable is False

    def test_413_names_the_setting_that_fixes_it(self):
        failure = describe_llm_failure(ProviderError("too large", 413))
        assert "LLM_MAX_TOKENS" in failure.message


class TestToolUseFailed:
    """`tool_use_failed` is the model missing a tool call, not a bad request.

    Groq reports it as a 400, and treating every 400 as a client error meant
    the agent gave up after one attempt on exactly the transient failure the
    retry budget exists for. The observed case had an empty `failed_generation`
    — the model produced nothing at all.
    """

    @staticmethod
    def _tool_use_failed():
        return ProviderError(
            "Error code: 400 - {'error': {'message': 'Tool choice is required, "
            "but model did not call a tool', 'code': 'tool_use_failed', "
            "'failed_generation': ''}}",
            400,
        )

    def test_it_is_retried(self):
        assert describe_llm_failure(self._tool_use_failed()).retryable is True

    def test_it_names_the_settings_that_help(self):
        message = describe_llm_failure(self._tool_use_failed()).message
        assert "LLM_REASONING_EFFORT" in message

    def test_a_genuinely_malformed_request_is_still_not_retried(self):
        """Not every 400 is worth another attempt — the distinction is the
        point of the change."""
        failure = describe_llm_failure(ProviderError("unsupported field 'x'", 400))
        assert failure.retryable is False

    async def test_the_agent_uses_its_full_retry_budget(self, wired, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            graph,
            "get_structured_llm",
            always_raising(self._tool_use_failed(), calls),
        )

        final = await graph.run_agent("u1", today=TODAY)

        meal_errors = sum(
            1
            for s in final["steps"]
            if s["node"] == "plan_meals" and s["status"] == "error"
        )
        assert meal_errors == graph.MAX_GENERATION_ATTEMPTS, (
            "a missed tool call should be retried, not treated as a bad request"
        )


class TestReasoningEffort:
    def test_it_is_omitted_when_unset(self, monkeypatch):
        """Providers that do not know the field must be unaffected."""
        import sys
        import types

        import app.agent.llm as llm_module

        captured = {}

        class FakeChat:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("langchain_groq")
        fake.ChatGroq = FakeChat
        monkeypatch.setitem(sys.modules, "langchain_groq", fake)
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_reasoning_effort", None)
        llm_module.reset_cache()

        llm_module.get_llm(1000)
        assert captured["model_kwargs"] == {}
        llm_module.reset_cache()

    def test_it_is_passed_through_when_set(self, monkeypatch):
        import sys
        import types

        import app.agent.llm as llm_module

        captured = {}

        class FakeChat:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("langchain_groq")
        fake.ChatGroq = FakeChat
        monkeypatch.setitem(sys.modules, "langchain_groq", fake)
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_reasoning_effort", "low")
        llm_module.reset_cache()

        llm_module.get_llm(1000)
        assert captured["model_kwargs"] == {"reasoning_effort": "low"}
        llm_module.reset_cache()


class TestReasoningEffortSetting:
    def test_blank_means_unset(self):
        from app.core.config import Settings

        import os

        os.environ["LLM_REASONING_EFFORT"] = ""
        try:
            assert Settings(_env_file=None).llm_reasoning_effort is None
        finally:
            os.environ.pop("LLM_REASONING_EFFORT", None)

    def test_case_is_normalised(self):
        from app.core.config import Settings

        import os

        os.environ["LLM_REASONING_EFFORT"] = "  LOW "
        try:
            assert Settings(_env_file=None).llm_reasoning_effort == "low"
        finally:
            os.environ.pop("LLM_REASONING_EFFORT", None)

    def test_a_nonsense_value_is_rejected_at_startup(self):
        """Better to fail loudly on boot than to have every model call 400."""
        import os

        from app.core.config import Settings

        os.environ["LLM_REASONING_EFFORT"] = "maximum"
        try:
            with pytest.raises(Exception, match="low, medium or high"):
                Settings(_env_file=None)
        finally:
            os.environ.pop("LLM_REASONING_EFFORT", None)


class TestJsonModeEscapeHatch:
    """`tool_use_failed` is a tool-calling failure specifically.

    json_mode asks for raw JSON instead, so it sidesteps the mechanism that is
    failing. The provider then only guarantees valid JSON, not the right shape,
    so the schema has to travel in the prompt.
    """

    @staticmethod
    def _adapter(schema):
        import app.agent.llm as llm_module

        class Captor:
            sent = None

            async def ainvoke(self, messages):
                Captor.sent = messages
                return "parsed"

        captor = Captor()
        return llm_module._JsonModeAdapter(captor, schema), Captor

    async def test_the_schema_is_appended_to_the_prompt(self):
        from langchain_core.messages import HumanMessage

        from app.models.plan import TrainingPlanDraft

        adapter, captor = self._adapter(TrainingPlanDraft)
        await adapter.ainvoke([HumanMessage(content="Plan a week.")])

        sent = captor.sent[-1].content
        assert "Plan a week." in sent, "the original prompt must survive"
        assert "activity_type" in sent, "the schema must reach the model"

    async def test_the_original_message_is_not_mutated(self):
        """The caller's message object is reused across retries."""
        from langchain_core.messages import HumanMessage

        from app.models.plan import TrainingPlanDraft

        original = HumanMessage(content="Plan a week.")
        adapter, _ = self._adapter(TrainingPlanDraft)
        await adapter.ainvoke([original])

        assert original.content == "Plan a week."

    async def test_a_plain_string_prompt_also_works(self):
        from app.models.plan import TrainingPlanDraft

        adapter, captor = self._adapter(TrainingPlanDraft)
        await adapter.ainvoke("Plan a week.")

        assert "Plan a week." in captor.sent
        assert "activity_type" in captor.sent

    def test_an_unknown_method_is_rejected_at_startup(self):
        import os

        from app.core.config import Settings

        os.environ["LLM_STRUCTURED_METHOD"] = "telepathy"
        try:
            with pytest.raises(Exception, match="function_calling or json_mode"):
                Settings(_env_file=None)
        finally:
            os.environ.pop("LLM_STRUCTURED_METHOD", None)

    def test_the_default_is_function_calling(self):
        from app.core.config import Settings

        assert Settings(_env_file=None).llm_structured_method == "function_calling"


class TestJsonValidateFailed:
    """The json_mode counterpart of a missed tool call.

    In json_mode nothing constrains the output, so a long nested array is where
    a model drifts: the observed failure closed day 1 correctly and then wrote
    `,"day":2` straight into the array without opening a brace. Groq reports
    that as a 400, which was classified non-retryable — so the run gave up
    after a single attempt at a failure that is different every time.
    """

    @staticmethod
    def _json_validate_failed():
        return ProviderError(
            "Error code: 400 - {'error': {'message': \"Failed to generate JSON. "
            "Please adjust your prompt. See 'failed_generation' for more "
            "details.\", 'code': 'json_validate_failed'}}",
            400,
        )

    def test_it_is_retried(self):
        assert describe_llm_failure(self._json_validate_failed()).retryable is True

    def test_a_genuinely_malformed_request_is_still_not_retried(self):
        assert (
            describe_llm_failure(ProviderError("unsupported field", 400)).retryable
            is False
        )

    async def test_the_agent_uses_its_full_retry_budget(self, wired, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            graph,
            "get_structured_llm",
            always_raising(self._json_validate_failed(), calls),
        )

        final = await graph.run_agent("u1", today=TODAY)

        meal_errors = sum(
            1
            for s in final["steps"]
            if s["node"] == "plan_meals" and s["status"] == "error"
        )
        assert meal_errors == graph.MAX_GENERATION_ATTEMPTS


class TestTrainerOutputSize:
    """Every field the trainer emits is a field it can get wrong.

    The trainer's JSON started drifting once each day carried a list of
    exercises. `cue` was the worst offender: filled from the exercise table
    during assembly anyway, so asking for it produced ~35 `"cue": null` per
    week — output that had to be emitted correctly for no benefit, in exactly
    the place the drift happened.
    """

    def test_the_draft_does_not_ask_for_cues(self):
        from app.models.plan import ExerciseDraft

        assert "cue" not in ExerciseDraft.model_fields

    def test_the_draft_does_not_ask_for_step_targets(self):
        """Seven identical values with a sensible default."""
        from app.models.plan import ActivityDraft

        assert "target_steps" not in ActivityDraft.model_fields

    def test_the_stored_form_still_has_both(self):
        """Narrowing what the model emits must not narrow what we keep."""
        from app.models.plan import ActivityItem, ExercisePrescription

        assert "cue" in ExercisePrescription.model_fields
        assert "target_steps" in ActivityItem.model_fields

    def test_widening_preserves_the_session(self):
        from app.models.plan import ActivityDraft, ExerciseDraft

        draft = ActivityDraft(
            activity_type="Strength training — full body",
            duration_minutes=45,
            intensity="moderate",
            description="Compound work.",
            exercises=[ExerciseDraft(name="Push-ups", sets=3, reps="8-12")],
        )
        item = draft.to_activity_item()

        assert item.activity_type == draft.activity_type
        assert [e.name for e in item.exercises] == ["Push-ups"]
        assert item.target_steps > 0, "the default should apply"


class TestRateLimitWaiting:
    """Being rate limited is not a failure of the plan.

    A 7-day plan costs most of an 8000 TPM window on its first attempt, so an
    immediate retry is guaranteed to be refused — the observed run spent
    attempts 2 and 3 on nothing but 429s. Waiting for the window to roll keeps
    the generation budget for problems that are actually about the food.
    """

    class Waits:
        """Records sleeps instead of taking them."""

        def __init__(self):
            self.slept: list[float] = []

        async def __call__(self, seconds: float) -> None:
            self.slept.append(seconds)

    @staticmethod
    def _rate_limited(hint: str = ""):
        return ProviderError(f"Rate limit reached. {hint}", 429)

    def _wrap(self, inner, waits):
        import app.agent.llm as llm_module

        return llm_module._RateLimitRetrying(inner, sleep=waits)

    async def test_it_waits_and_succeeds(self):
        from app.agent.llm import _RateLimitRetrying  # noqa: F401

        waits = self.Waits()
        calls = {"n": 0}

        class Flaky:
            async def ainvoke(self, _messages):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise TestRateLimitWaiting._rate_limited("Please try again in 22.5s")
                return "the plan"

        result = await self._wrap(Flaky(), waits).ainvoke([])

        assert result == "the plan"
        assert waits.slept == [22.5], "the provider's own number should be used"

    async def test_it_falls_back_when_no_hint_is_given(self):
        from app.agent.llm import DEFAULT_RATE_LIMIT_WAIT_SECONDS

        waits = self.Waits()
        calls = {"n": 0}

        class Flaky:
            async def ainvoke(self, _messages):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise TestRateLimitWaiting._rate_limited()
                return "the plan"

        await self._wrap(Flaky(), waits).ainvoke([])
        assert waits.slept == [DEFAULT_RATE_LIMIT_WAIT_SECONDS]

    async def test_it_gives_up_eventually(self):
        from app.agent.llm import MAX_RATE_LIMIT_WAITS

        waits = self.Waits()

        class AlwaysLimited:
            async def ainvoke(self, _messages):
                raise TestRateLimitWaiting._rate_limited()

        with pytest.raises(ProviderError):
            await self._wrap(AlwaysLimited(), waits).ainvoke([])

        assert len(waits.slept) == MAX_RATE_LIMIT_WAITS

    async def test_other_failures_are_not_waited_on(self):
        """A retired model does not become available in thirty seconds."""
        waits = self.Waits()

        class NotFound:
            async def ainvoke(self, _messages):
                raise ProviderError("model_not_found", 404)

        with pytest.raises(ProviderError):
            await self._wrap(NotFound(), waits).ainvoke([])

        assert waits.slept == []

    async def test_a_successful_call_never_waits(self):
        waits = self.Waits()

        class Fine:
            async def ainvoke(self, _messages):
                return "the plan"

        assert await self._wrap(Fine(), waits).ainvoke([]) == "the plan"
        assert waits.slept == []

    def test_the_wait_is_capped(self):
        """A provider asking for ten minutes should not freeze the run."""
        from app.agent.llm import (
            MAX_RATE_LIMIT_WAIT_SECONDS,
            parse_retry_after_seconds,
        )

        assert parse_retry_after_seconds("try again in 10m0s") == (
            MAX_RATE_LIMIT_WAIT_SECONDS
        )

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Please try again in 1m13.5s", 73.5),
            ("try again in 24.9s", 24.9),
            ("nothing useful here", None),
        ],
    )
    def test_hint_parsing(self, text, expected):
        from app.agent.llm import parse_retry_after_seconds

        assert parse_retry_after_seconds(text) == expected


class TestBudgetsScaleWithPlanLength:
    """A flat budget is wrong in both directions.

    Sized for a seven-day plan it over-reserves for a four-day one, and
    providers charge reserved output against the per-minute limit whether it is
    used or not. The trainer was reserving 1400 tokens to write four days.
    """

    def test_a_shorter_plan_reserves_less(self, monkeypatch):
        import app.agent.graph as graph_module
        from app.agent.llm import budget_for
        from app.models.plan import TrainingPlanDraft

        monkeypatch.setattr(graph_module, "PLAN_DURATION_DAYS", 7)
        seven = budget_for(TrainingPlanDraft)

        monkeypatch.setattr(graph_module, "PLAN_DURATION_DAYS", 3)
        three = budget_for(TrainingPlanDraft)

        assert three < seven

    def test_the_meal_budget_follows_the_chunk_not_the_plan(self, monkeypatch):
        """The nutritionist drafts a chunk at a time, so a longer plan means
        more calls, not a bigger one."""
        import app.agent.graph as graph_module
        from app.agent.llm import budget_for
        from app.models.plan import MealPlanDraft

        monkeypatch.setattr(graph_module, "PLAN_DURATION_DAYS", 7)
        seven = budget_for(MealPlanDraft)

        monkeypatch.setattr(
            graph_module, "PLAN_DURATION_DAYS", graph_module.MEAL_CHUNK_DAYS
        )
        one_chunk = budget_for(MealPlanDraft)

        assert seven == one_chunk

    def test_a_one_day_plan_does_not_reserve_a_week(self, monkeypatch):
        import app.agent.graph as graph_module
        from app.agent.llm import budget_for
        from app.models.plan import MealPlanDraft

        monkeypatch.setattr(graph_module, "PLAN_DURATION_DAYS", 1)
        assert budget_for(MealPlanDraft) < 1200

    def test_the_ceiling_still_applies(self, monkeypatch):
        from app.agent.llm import budget_for
        from app.core.config import settings
        from app.models.plan import MealPlanDraft

        monkeypatch.setattr(settings, "llm_max_tokens", 700)
        assert budget_for(MealPlanDraft) == 700

    def test_every_budget_leaves_room_for_the_answer(self, monkeypatch):
        """Scaling must not shrink a budget below what the output needs.

        Roughly 70 tokens a meal, four meals a day, plus the reasoning field.
        """
        import app.agent.graph as graph_module
        from app.agent.llm import budget_for
        from app.models.plan import MealPlanDraft

        for days in (1, 3, 4, 7):
            monkeypatch.setattr(graph_module, "PLAN_DURATION_DAYS", days)
            chunk = min(graph_module.MEAL_CHUNK_DAYS, days)
            needed = chunk * 4 * 70 + 150

            assert budget_for(MealPlanDraft) > needed, (
                f"{days}-day plan: budget too tight for the meals it must hold"
            )
