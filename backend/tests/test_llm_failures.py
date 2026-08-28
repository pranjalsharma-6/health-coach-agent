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
    def factory(_schema, **_budget):
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
        # A key has to be present for provider resolution to name a model.
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_provider", None)
        monkeypatch.setattr(settings, "llm_model", "")

        calls: list = []
        monkeypatch.setattr(
            graph,
            "get_structured_llm",
            always_raising(ProviderError("model_not_found", 404), calls),
        )

        final = await graph.run_agent("u1", today=TODAY)
        messages = " ".join(s["message"] for s in final["steps"])

        from app.agent.llm import resolve_model, resolve_provider

        model = resolve_model(resolve_provider() or "groq")
        assert model.lower() in messages.lower(), (
            "the resolved model should be named — LLM_MODEL is blank when a "
            "provider default is in use, and naming '' helps nobody"
        )
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

        def factory(schema, **_budget):
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
        from app.agent.llm import FIXED_OUTPUT_TOKENS

        for name, budget in FIXED_OUTPUT_TOKENS.items():
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

        # A floor rather than the whole figure: every budget carries room for
        # the model to think, which is charged to the same reservation.
        assert budget_for(SomethingNew) >= DEFAULT_MAX_OUTPUT_TOKENS

    def test_llm_max_tokens_caps_every_budget(self, monkeypatch):
        """The escape hatch for a key on a tighter limit than the defaults
        assume."""
        from app.agent.llm import budget_for
        from app.core.config import settings
        from app.models.plan import MealPlanDraft, PlanCritique

        monkeypatch.setattr(settings, "llm_max_tokens", 1500)

        assert budget_for(MealPlanDraft) == 1500
        # A budget already below the ceiling is left alone rather than raised.
        assert budget_for(PlanCritique) < 1500

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

    def test_413_points_at_a_smaller_plan_not_a_smaller_budget(self):
        """It used to say "try LLM_MAX_TOKENS=1500".

        That number is below what a plan needs to finish, so following the
        advice turned a loud 413 into a plan silently cut off mid-week — the
        harder failure to recognise, reached by doing what the error said. The
        honest fix for a request that is too large is to ask for less plan.
        """
        message = describe_llm_failure(ProviderError("too large", 413)).message

        assert "PLAN_DURATION_DAYS" in message
        assert "LLM_MAX_TOKENS" not in message


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

    def test_it_does_not_repeat_advice_the_user_has_already_taken(self):
        """It used to say "set LLM_REASONING_EFFORT=low" — to a user who had.

        The budget now carries its own room to think, so that setting is no
        longer the fix. Telling someone to do what they have already done makes
        a fault in the code read as a mistake of theirs, and sends them round
        the same loop.
        """
        message = describe_llm_failure(self._tool_use_failed()).message

        assert "LLM_REASONING_EFFORT" not in message
        assert "reserves more" in message, (
            "the message should say what changes on the next attempt, since "
            "something does"
        )

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
        one_day = budget_for(MealPlanDraft)

        monkeypatch.setattr(graph_module, "PLAN_DURATION_DAYS", 4)
        four_days = budget_for(MealPlanDraft)

        # Relative, not an absolute ceiling: every budget also carries a fixed
        # allowance for scaffolding and reasoning, so the right question is
        # whether a shorter plan costs less, not whether it fits a magic number.
        assert one_day < four_days

    def test_the_ceiling_still_applies(self, monkeypatch):
        from app.agent.llm import budget_for
        from app.core.config import settings
        from app.models.plan import MealPlanDraft

        monkeypatch.setattr(settings, "llm_max_tokens", 1500)
        assert budget_for(MealPlanDraft) == 1500

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


class TestDailyVersusPerMinuteLimits:
    """Two completely different problems behind one status code.

    A per-minute cap clears by waiting; a daily allowance does not. The
    original message told a user whose day's tokens were spent that "waiting a
    minute usually clears it", which sent them round the same loop three times.
    """

    DAILY = (
        "Rate limit reached for model `openai/gpt-oss-120b` in organization "
        "`org_x` service tier `on_demand` on tokens per day (TPD): Limit "
        "200000, Used 197032, Requested 4220. Please try again in 9m0.864s."
    )
    PER_MINUTE = (
        "Rate limit reached on tokens per minute (TPM): Limit 8000, "
        "Requested 9243. Please try again in 24.9s."
    )

    @staticmethod
    def _limited(message):
        return ProviderError(message, 429)

    def test_a_daily_limit_is_recognised(self):
        from app.agent.llm import is_daily_limit

        assert is_daily_limit(self.DAILY)
        assert not is_daily_limit(self.PER_MINUTE)

    def test_a_daily_limit_with_a_long_wait_is_not_retried(self):
        """Three attempts against a spent daily allowance is three failures."""
        assert describe_llm_failure(self._limited(self.DAILY)).retryable is False

    def test_a_per_minute_limit_is_retried(self):
        assert describe_llm_failure(self._limited(self.PER_MINUTE)).retryable is True

    def test_the_daily_message_does_not_suggest_waiting_a_minute(self):
        message = describe_llm_failure(self._limited(self.DAILY)).message

        assert "day" in message.lower()
        assert "waiting a minute" not in message.lower()

    def test_the_daily_message_names_what_would_actually_help(self):
        message = describe_llm_failure(self._limited(self.DAILY)).message

        assert "PLAN_DURATION_DAYS" in message

    def test_the_provider_s_own_estimate_is_quoted(self):
        """Nine minutes, from the error text — not a number we invented."""
        message = describe_llm_failure(self._limited(self.DAILY)).message

        assert "9 minutes" in message

    async def test_the_waiter_does_not_sit_on_a_long_wait(self):
        """Blocking a request for nine minutes is not a retry, it is a hang."""
        import app.agent.llm as llm_module

        slept: list = []

        async def record(seconds):
            slept.append(seconds)

        class DailyLimited:
            async def ainvoke(self, _messages):
                raise TestDailyVersusPerMinuteLimits._limited(
                    TestDailyVersusPerMinuteLimits.DAILY
                )

        with pytest.raises(ProviderError):
            await llm_module._RateLimitRetrying(DailyLimited(), sleep=record).ainvoke([])

        assert slept == [], "a daily limit must fail fast, not wait"

    async def test_the_waiter_still_waits_on_a_per_minute_limit(self):
        import app.agent.llm as llm_module

        slept: list = []

        async def record(seconds):
            slept.append(seconds)

        calls = {"n": 0}

        class Flaky:
            async def ainvoke(self, _messages):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise TestDailyVersusPerMinuteLimits._limited(
                        TestDailyVersusPerMinuteLimits.PER_MINUTE
                    )
                return "the plan"

        result = await llm_module._RateLimitRetrying(Flaky(), sleep=record).ainvoke([])

        assert result == "the plan"
        assert slept == [24.9]

    def test_an_uncapped_read_is_available_for_reporting(self):
        """The wait is capped for *waiting* — telling the user it is nine
        minutes should not report ninety seconds."""
        from app.agent.llm import parse_retry_after_seconds

        assert parse_retry_after_seconds(self.DAILY) == 90.0  # capped, for sleeping
        assert parse_retry_after_seconds(self.DAILY, cap=None) == 540.864


class TestProviderSelection:
    """Three providers, one abstraction.

    Groq's free tier caps daily tokens at a level a multi-agent planner reaches
    quickly. Being able to move to another provider is a config change here
    precisely because nothing outside this module names a vendor.
    """

    @staticmethod
    def _only(monkeypatch, **keys):
        import app.agent.llm as llm_module

        for field in ("groq_api_key", "gemini_api_key", "openai_api_key"):
            monkeypatch.setattr(settings, field, keys.get(field, ""))
        monkeypatch.setattr(settings, "llm_provider", keys.get("provider"))
        monkeypatch.setattr(settings, "llm_model", keys.get("model", ""))
        llm_module.reset_cache()

    @pytest.mark.parametrize(
        "field,expected",
        [
            ("groq_api_key", "groq"),
            ("gemini_api_key", "gemini"),
            ("openai_api_key", "openai"),
        ],
    )
    def test_one_key_needs_no_provider_setting(self, monkeypatch, field, expected):
        """Asking someone with one key to also name the provider is a second
        chance to get it wrong."""
        from app.agent.llm import resolve_provider

        self._only(monkeypatch, **{field: "k"})
        assert resolve_provider() == expected

    def test_an_explicit_provider_wins(self, monkeypatch):
        from app.agent.llm import resolve_provider

        self._only(
            monkeypatch, groq_api_key="g", gemini_api_key="k", provider="gemini"
        )
        assert resolve_provider() == "gemini"

    def test_no_key_resolves_to_nothing(self, monkeypatch):
        from app.agent.llm import is_configured, resolve_provider

        self._only(monkeypatch)
        assert resolve_provider() is None
        assert not is_configured()

    def test_a_key_without_its_provider_is_not_configured(self, monkeypatch):
        """LLM_PROVIDER=gemini with only a Groq key set is a misconfiguration,
        not a silent fallback to Groq."""
        from app.agent.llm import is_configured

        self._only(monkeypatch, groq_api_key="g", provider="gemini")
        assert not is_configured()

    @pytest.mark.parametrize(
        "provider,fragment",
        [("groq", "gpt-oss"), ("gemini", "gemini"), ("openai", "gpt-4o")],
    )
    def test_each_provider_has_its_own_default_model(self, provider, fragment):
        """A Groq model name means nothing to Gemini. Sharing one default was
        how a provider switch silently sent the wrong name."""
        from app.agent.llm import resolve_model

        assert fragment in resolve_model(provider)

    def test_llm_model_overrides_the_default(self, monkeypatch):
        from app.agent.llm import resolve_model

        self._only(monkeypatch, gemini_api_key="k", model="gemini-2.0-flash")
        assert resolve_model("gemini") == "gemini-2.0-flash"

    def test_gemini_is_built_with_its_own_parameter_names(self, monkeypatch):
        """Gemini spells the output cap `max_output_tokens`, and takes the key
        as `google_api_key`. Passing Groq's names would fail at construction."""
        import sys
        import types

        import app.agent.llm as llm_module

        captured = {}

        class FakeGemini:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("langchain_google_genai")
        fake.ChatGoogleGenerativeAI = FakeGemini
        monkeypatch.setitem(sys.modules, "langchain_google_genai", fake)

        self._only(monkeypatch, gemini_api_key="AIza-test")
        llm_module.get_llm(1500)

        assert captured["max_output_tokens"] == 1500
        assert captured["google_api_key"] == "AIza-test"
        assert "gemini" in captured["model"]
        llm_module.reset_cache()

    def test_gemini_is_not_sent_groq_only_parameters(self, monkeypatch):
        """`reasoning_effort` is a Groq field. Sending it to Gemini is a 400."""
        import sys
        import types

        import app.agent.llm as llm_module

        captured = {}

        class FakeGemini:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake = types.ModuleType("langchain_google_genai")
        fake.ChatGoogleGenerativeAI = FakeGemini
        monkeypatch.setitem(sys.modules, "langchain_google_genai", fake)

        self._only(monkeypatch, gemini_api_key="AIza-test")
        monkeypatch.setattr(settings, "llm_reasoning_effort", "low")
        llm_module.get_llm(1500)

        assert "model_kwargs" not in captured
        assert "reasoning_effort" not in captured
        llm_module.reset_cache()

    def test_an_unknown_provider_is_rejected_at_startup(self):
        import os

        from app.core.config import Settings

        os.environ["LLM_PROVIDER"] = "skynet"
        try:
            with pytest.raises(Exception, match="groq, openai or gemini"):
                Settings(_env_file=None)
        finally:
            os.environ.pop("LLM_PROVIDER", None)


class TestRetiredModels:
    """A hardcoded model default is a default that goes stale.

    `gemini-2.5-flash` was current when it was written into this repo and
    refused for new keys by the time it ran. The fix is not a better guess: it
    is to repeat what the provider says, since the provider is the authority on
    its own model names.
    """

    GOOGLE_404 = (
        "404 This model models/gemini-2.5-flash is no longer available to new "
        "users. Please update your code to use models/gemini-3.6-flash for the "
        "latest features and improvements. We recommend you to use the "
        "Interactions API."
    )

    def test_the_replacement_is_extracted(self):
        from app.agent.llm import suggested_replacement

        assert suggested_replacement(self.GOOGLE_404) == "gemini-3.6-flash"

    def test_prose_is_not_mistaken_for_a_model(self):
        """"use the Interactions API" is advice, not a model name. Without a
        digit test it is exactly what the pattern would grab."""
        from app.agent.llm import suggested_replacement

        assert suggested_replacement("We recommend you use the Interactions API") is None
        assert suggested_replacement("please use a supported model") is None

    def test_a_bare_404_offers_the_listing_tool_instead(self):
        failure = describe_llm_failure(ProviderError("model_not_found", 404))

        assert "check_llm" in failure.message
        assert failure.retryable is False

    def test_the_suggestion_is_turned_into_an_instruction(self):
        failure = describe_llm_failure(ProviderError(self.GOOGLE_404, 404))

        assert "LLM_MODEL=gemini-3.6-flash" in failure.message
        assert failure.retryable is False

    def test_it_is_never_retried(self):
        """A retired model does not come back between attempts."""
        assert describe_llm_failure(ProviderError(self.GOOGLE_404, 404)).retryable is False

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("please use gpt-4o instead", "gpt-4o"),
            ("update your code to use models/gemini-3.6-flash", "gemini-3.6-flash"),
            ("use claude-sonnet-5.", "claude-sonnet-5"),
            ("nothing suggested here", None),
        ],
    )
    def test_extraction_across_provider_phrasings(self, text, expected):
        from app.agent.llm import suggested_replacement

        assert suggested_replacement(text) == expected


class TestGeminiStructuredOutput:
    """`method="json_mode"` is an OpenAI-style option Gemini does not have.

    Its `with_structured_output` takes **kwargs and drops unknown ones, so the
    setting was accepted and ignored — while the adapter still appended a full
    JSON Schema to the prompt. Gemini then had its own function-calling path
    *and* a schema dump telling it to return raw JSON. The reply could not be
    coerced, LangChain's parser returned None, and the None travelled until
    `draft.days` raised AttributeError in the graph.
    """

    def _fake_gemini(self, monkeypatch, captured):
        import sys
        import types

        class FakeStructured:
            async def ainvoke(self, _messages):
                return "structured"

        class FakeGemini:
            def __init__(self, **kwargs):
                pass

            def with_structured_output(self, schema, **kwargs):
                captured.update(kwargs)
                return FakeStructured()

        fake = types.ModuleType("langchain_google_genai")
        fake.ChatGoogleGenerativeAI = FakeGemini
        monkeypatch.setitem(sys.modules, "langchain_google_genai", fake)

        for field in ("groq_api_key", "openai_api_key"):
            monkeypatch.setattr(settings, field, "")
        monkeypatch.setattr(settings, "gemini_api_key", "AIza-test")
        monkeypatch.setattr(settings, "llm_provider", "gemini")

    def test_json_mode_is_not_requested_from_gemini(self, monkeypatch):
        import app.agent.llm as llm_module
        from app.models.plan import TrainingPlanDraft

        captured: dict = {}
        self._fake_gemini(monkeypatch, captured)
        monkeypatch.setattr(settings, "llm_structured_method", "json_mode")
        llm_module.reset_cache()

        llm_module.get_structured_llm(TrainingPlanDraft)

        assert "method" not in captured, (
            "Gemini drops the kwarg silently; asking for it only misleads"
        )
        llm_module.reset_cache()

    async def test_gemini_is_not_sent_a_schema_dump(self, monkeypatch):
        """The adapter's prompt injection belongs to json_mode only."""
        import app.agent.llm as llm_module
        from langchain_core.messages import HumanMessage
        from app.models.plan import TrainingPlanDraft

        captured: dict = {}
        self._fake_gemini(monkeypatch, captured)
        monkeypatch.setattr(settings, "llm_structured_method", "json_mode")
        llm_module.reset_cache()

        runnable = llm_module.get_structured_llm(TrainingPlanDraft)
        assert not isinstance(runnable, llm_module._JsonModeAdapter)
        llm_module.reset_cache()

    @pytest.mark.parametrize("provider,expected", [
        ("groq", True), ("openai", True), ("gemini", False),
    ])
    def test_which_providers_support_the_option(self, monkeypatch, provider, expected):
        from app.agent.llm import provider_supports_json_mode

        monkeypatch.setattr(settings, "llm_provider", provider)
        assert provider_supports_json_mode() is expected


class TestNeverNone:
    """A structured parser that returns None must not be allowed to travel.

    LangChain returns None when it cannot coerce a response — no exception, no
    message. The observed traceback pointed at `draft.days` in the graph, four
    frames and one file away from the model call that produced nothing.
    """

    @staticmethod
    def _wrap(result):
        import app.agent.llm as llm_module
        from app.models.plan import TrainingPlanDraft

        class Returns:
            async def ainvoke(self, _messages):
                return result

        return llm_module._NeverNone(Returns(), TrainingPlanDraft)

    async def test_none_raises_where_it_happens(self):
        with pytest.raises(ValueError, match="returned nothing usable"):
            await self._wrap(None).ainvoke([])

    async def test_the_error_names_the_schema(self):
        with pytest.raises(ValueError, match="TrainingPlanDraft"):
            await self._wrap(None).ainvoke([])

    async def test_a_real_result_passes_through(self):
        assert await self._wrap("a draft").ainvoke([]) == "a draft"

    async def test_falsy_but_present_results_are_not_rejected(self):
        """Only None means "could not parse". An empty list is an answer."""
        assert await self._wrap([]).ainvoke([]) == []

    def test_the_failure_is_retryable(self):
        """An unparseable reply is the transient case the retry budget exists
        for — unlike a retired model or a bad key."""
        assert describe_llm_failure(ValueError("returned nothing usable")).retryable


class TestProviderExceptionShapes:
    """Providers spell the same failure differently.

    Groq and OpenAI raise exceptions with `status_code`. Google's `api_core`
    exceptions carry `code`. A classifier that only knew `status_code` was
    blind to every Gemini rate limit — the run retried instantly three times,
    with no wait and a generic message, because the 429 was never recognised
    as a 429.
    """

    class GoogleStyle(Exception):
        """Shaped like google.api_core.exceptions.ResourceExhausted."""

        def __init__(self, message, code=429):
            super().__init__(message)
            self.code = code

    GOOGLE_QUOTA = (
        "429 You exceeded your current quota. * Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 20, model: gemini-3.6-flash Please retry in 31.24s. "
        'violations { quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier" }'
    )

    def test_a_google_style_exception_is_recognised(self):
        from app.agent.llm import http_status

        assert http_status(self.GoogleStyle("quota")) == 429

    def test_an_openai_style_exception_is_recognised(self):
        from app.agent.llm import http_status

        assert http_status(ProviderError("limited", 429)) == 429

    def test_the_status_is_read_from_the_text_as_a_last_resort(self):
        """Some wrappers expose neither attribute."""
        from app.agent.llm import http_status

        assert http_status(RuntimeError("got a 503 from upstream")) == 503

    def test_a_plain_exception_has_no_status(self):
        from app.agent.llm import http_status

        assert http_status(ValueError("could not parse")) is None

    def test_a_year_is_not_mistaken_for_a_status(self):
        from app.agent.llm import http_status

        assert http_status(ValueError("failed at 2026-08-28")) is None

    def test_googles_quota_error_is_classified_and_waited_on(self):
        """It names a daily quota *and* a 31-second wait. The wait is what
        matters — refusing to wait throws away a run half a minute from
        working."""
        failure = describe_llm_failure(self.GoogleStyle(self.GOOGLE_QUOTA))

        assert failure.retryable is True
        assert "31 seconds" in failure.message

    def test_googles_retry_phrasing_is_parsed(self):
        """Groq says "try again in", Google says "retry in"."""
        from app.agent.llm import parse_retry_after_seconds

        assert parse_retry_after_seconds("Please retry in 31.24s") == 31.24
        assert parse_retry_after_seconds("Please try again in 24.9s") == 24.9

    def test_googles_quota_id_reads_as_a_daily_limit(self):
        from app.agent.llm import is_daily_limit

        assert is_daily_limit(self.GOOGLE_QUOTA)

    async def test_the_waiter_now_sees_google_rate_limits(self):
        import app.agent.llm as llm_module

        slept: list = []

        async def record(seconds):
            slept.append(seconds)

        calls = {"n": 0}
        outer = self

        class Flaky:
            async def ainvoke(self, _messages):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise outer.GoogleStyle(outer.GOOGLE_QUOTA)
                return "the plan"

        result = await llm_module._RateLimitRetrying(Flaky(), sleep=record).ainvoke([])

        assert result == "the plan"
        assert slept == [31.24], "it should wait exactly as long as Google asked"


class TestModelBelongsToProvider:
    """A model name from one provider must not be sent to another.

    This is the bug that produced a bare 404 from `api.groq.com` for
    `gemini-3.6-flash`: `LLM_MODEL` was applied to whichever provider happened
    to be active, and nothing compared the two. The error named the model but
    not the endpoint, so it read as a retired model rather than as one line in
    `.env` left over from trying a different vendor.
    """

    @pytest.mark.parametrize(
        "name,owner",
        [
            ("gemini-3.6-flash", "gemini"),
            ("models/gemini-2.5-flash", "gemini"),
            ("gemini-2.0-flash-exp", "gemini"),
            ("gpt-4o-mini", "openai"),
            ("gpt-4.1", "openai"),
            ("o3-mini", "openai"),
            ("llama-3.3-70b-versatile", "groq"),
            ("gemma2-9b-it", "groq"),
            ("qwen3-32b", "groq"),
            ("deepseek-r1-distill-llama-70b", "groq"),
        ],
    )
    def test_recognises_each_vendors_naming(self, name, owner):
        from app.agent.llm import model_belongs_to

        assert model_belongs_to(name) == owner

    @pytest.mark.parametrize(
        "name",
        ["openai/gpt-oss-120b", "meta-llama/llama-4-scout-17b-16e-instruct"],
    )
    def test_a_namespaced_id_is_groqs_convention_not_the_vendors(self, name):
        """`openai/gpt-oss-120b` is served by Groq, not by OpenAI.

        The prefix names who trained the model, not who hosts it. Reading it as
        a vendor would reject Groq's own default.
        """
        from app.agent.llm import model_belongs_to

        assert model_belongs_to(name) == "groq"

    @pytest.mark.parametrize("name", ["", "   ", "some-new-model-2027", "custom"])
    def test_an_unrecognised_name_is_passed_through(self, name):
        """Silence, not a guess.

        Providers ship new names constantly. Refusing everything this file has
        not heard of would break on the next release; the check exists to catch
        an obvious mistake, not to police the catalogue.
        """
        from app.agent.llm import model_belongs_to

        assert model_belongs_to(name) is None

    def test_a_matching_pair_is_not_a_problem(self):
        from app.agent.llm import describe_model_mismatch

        assert describe_model_mismatch("groq", "openai/gpt-oss-120b") is None
        assert describe_model_mismatch("gemini", "gemini-3.6-flash") is None

    def test_the_message_names_both_sides_and_both_fixes(self):
        from app.agent.llm import describe_model_mismatch

        message = describe_model_mismatch("groq", "gemini-3.6-flash")

        assert message is not None
        assert "gemini-3.6-flash" in message
        assert "Groq" in message, "the message must say where it was being sent"
        assert "LLM_MODEL" in message and "LLM_PROVIDER" in message, (
            "both ways out should be spelled, since which one is right depends "
            "on what the user meant"
        )

    @pytest.mark.anyio
    async def test_get_llm_refuses_before_calling_the_api(self, monkeypatch):
        """The client is never built, so no request is made and no retry spent."""
        from app.agent import llm as llm_module

        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_model", "gemini-3.6-flash")
        llm_module.reset_cache()

        with pytest.raises(llm_module.LLMUnavailableError) as caught:
            llm_module.get_llm()

        assert "gemini-3.6-flash" in str(caught.value)
        llm_module.reset_cache()

    @pytest.mark.anyio
    async def test_the_run_reports_it_once_rather_than_retrying(
        self, wired, monkeypatch
    ):
        """`LLMUnavailableError` already routes straight to the record step.

        A configuration error fails identically every time, so three attempts
        buy nothing but three minutes.
        """
        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_model", "gemini-3.6-flash")

        from app.agent import llm as llm_module

        llm_module.reset_cache()

        final = await graph.run_agent("u1", today=TODAY)
        messages = " ".join(s["message"] for s in final["steps"])

        assert "gemini-3.6-flash" in messages
        assert "Groq" in messages
        llm_module.reset_cache()

    def test_configuration_problem_is_quiet_when_nothing_is_wrong(self, monkeypatch):
        from app.agent.llm import configuration_problem

        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_model", "")

        assert configuration_problem() is None

    def test_configuration_problem_catches_a_named_provider_with_no_key(
        self, monkeypatch
    ):
        from app.agent.llm import configuration_problem

        monkeypatch.setattr(settings, "llm_provider", "gemini")
        monkeypatch.setattr(settings, "gemini_api_key", "")
        monkeypatch.setattr(settings, "llm_model", "")

        problem = configuration_problem()
        assert problem is not None and "GEMINI_API_KEY" in problem

    def test_the_404_message_names_the_provider_it_was_sent_to(self, monkeypatch):
        """Which endpoint refused is half the diagnosis.

        "The model 'gemini-3.6-flash' is not available" is puzzling; "not
        available on your Groq API key" explains itself.
        """
        from app.agent.llm import describe_llm_failure

        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_model", "some-retired-model-9")

        message = describe_llm_failure(ProviderError("model_not_found", 404)).message
        assert "Groq" in message
