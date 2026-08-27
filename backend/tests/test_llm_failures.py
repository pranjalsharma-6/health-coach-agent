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
from app.tools.check_llm import looks_like_a_chat_model
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

        assert len(calls) == 2, (
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
        from tests.factories import make_critique, make_training_draft

        state = {"meal_calls": 0}

        def factory(schema):
            class Stub:
                async def ainvoke(self, _messages):
                    if schema is TrainingPlanDraft:
                        return make_training_draft()
                    if schema is PlanCritique:
                        return make_critique()
                    state["meal_calls"] += 1
                    if state["meal_calls"] == 1:
                        raise ProviderError("upstream hiccup", 503)
                    return make_meal_draft(TARGETS)

            return Stub()

        monkeypatch.setattr(graph, "get_structured_llm", factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is not None, "a transient error should recover"
        assert state["meal_calls"] == 2


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
