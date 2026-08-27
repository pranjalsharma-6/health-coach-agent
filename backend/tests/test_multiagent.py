"""Tests for the multi-agent planning pipeline.

These check the claims the architecture actually makes — that the specialists
run concurrently, that a retry redraws only what needs redrawing, and that the
critic can never make an unsafe plan safe — rather than just that the graph
compiles.
"""

import asyncio
from datetime import date

import pytest

from app.agent import graph
from app.models.enums import AgentDecision, DietType
from app.models.plan import MealPlanDraft, PlanCritique, TrainingPlanDraft
from tests.factories import (
    make_critique,
    make_meal_draft,
    make_log,
    make_plan_in_db,
    make_profile,
    make_targets,
    make_training_draft,
)

TODAY = date(2026, 3, 15)
TARGETS = make_targets()


class Recorder:
    """A stub LLM that records what was asked of it and in what order."""

    def __init__(self, meals=None, training=None, critique=None, delay=0.0):
        self.meals = list(meals or [make_meal_draft(TARGETS)])
        self.training = training if training is not None else make_training_draft()
        self.critique = critique if critique is not None else make_critique()
        self.delay = delay

        self.calls: list[str] = []
        self.started: list[tuple[str, float]] = []
        self.finished: list[tuple[str, float]] = []
        self.meal_prompts: list[str] = []
        self.training_prompts: list[str] = []

    def factory(self, schema):
        recorder = self

        class Stub:
            async def ainvoke(self, messages):
                name = schema.__name__
                recorder.calls.append(name)
                recorder.started.append((name, asyncio.get_event_loop().time()))

                text = "\n".join(str(m.content) for m in messages)
                if schema is MealPlanDraft:
                    recorder.meal_prompts.append(text)
                elif schema is TrainingPlanDraft:
                    recorder.training_prompts.append(text)

                if recorder.delay:
                    await asyncio.sleep(recorder.delay)

                recorder.finished.append(
                    (name, asyncio.get_event_loop().time())
                )

                if schema is TrainingPlanDraft:
                    return recorder.training
                if schema is PlanCritique:
                    return recorder.critique
                return (
                    recorder.meals.pop(0)
                    if len(recorder.meals) > 1
                    else recorder.meals[0]
                )

        return Stub()


@pytest.fixture
def wired(monkeypatch):
    """Point the graph at in-memory repositories and a recording LLM."""
    saved: list = []

    class FakePlanRepo:
        @staticmethod
        async def get_active(_user_id):
            return None

        @staticmethod
        async def save_new_version(plan):
            plan.id = f"plan-{len(saved) + 1}"
            plan.version = len(saved) + 1
            saved.append(plan)
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
    return saved


def nodes_in(final) -> list[str]:
    return [s["node"] for s in final.get("steps", [])]


class TestPipelineShape:
    async def test_both_specialists_and_the_critic_run(self, wired, monkeypatch):
        recorder = Recorder()
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is not None
        assert set(recorder.calls) == {
            "MealPlanDraft",
            "TrainingPlanDraft",
            "PlanCritique",
        }
        trace = nodes_in(final)
        for node in ("plan_meals", "plan_training", "assemble", "critique", "validate"):
            assert node in trace, f"{node} missing from {trace}"

    async def test_specialists_run_concurrently(self, wired, monkeypatch):
        """The fan-out has to actually overlap, or it buys nothing.

        With a delay in each stub, sequential execution would take 2x the delay;
        concurrent execution overlaps, so the second starts before the first
        finishes.
        """
        recorder = Recorder(delay=0.15)
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        await graph.run_agent("u1", today=TODAY)

        starts = {name: t for name, t in recorder.started if name.endswith("Draft")}
        ends = {name: t for name, t in recorder.finished if name.endswith("Draft")}

        assert len(starts) == 2, "both specialists should have run"
        first_finish = min(ends.values())
        last_start = max(starts.values())
        assert last_start < first_finish, (
            "the second specialist started only after the first finished — "
            "the fan-out is running sequentially"
        )

    async def test_each_specialist_sees_only_its_own_brief(self, wired, monkeypatch):
        """Splitting the prompts is the point; leaking both back defeats it."""
        recorder = Recorder()
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        await graph.run_agent("u1", today=TODAY)

        meal_prompt = recorder.meal_prompts[0].lower()
        training_prompt = recorder.training_prompts[0].lower()

        # The nutritionist gets diet constraints and macro targets.
        assert "vegetarian" in meal_prompt
        assert "protein" in meal_prompt
        # The trainer gets neither — it is told never to prescribe food.
        assert "daily targets" not in training_prompt
        assert "allergies" not in training_prompt


class TestSelectiveRetry:
    async def test_validation_failure_redraws_only_the_meals(
        self, wired, monkeypatch
    ):
        """The validator inspects food, so a rejection is about food.

        Re-running the trainer would spend a call to churn a week that was fine.
        """
        bad = make_meal_draft(TARGETS)
        bad.days[0].meals[0].name = "Grilled chicken salad"  # not vegetarian
        good = make_meal_draft(TARGETS)

        recorder = Recorder(meals=[bad, good])
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is not None
        assert recorder.calls.count("MealPlanDraft") == 2, "meals should be redrawn"
        assert recorder.calls.count("TrainingPlanDraft") == 1, (
            "training should be reused, not regenerated"
        )

    async def test_gives_up_after_the_attempt_budget(self, wired, monkeypatch):
        bad = make_meal_draft(TARGETS)
        bad.days[0].meals[0].name = "Grilled chicken salad"

        recorder = Recorder(meals=[bad])
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is None
        assert final["error"] is not None
        assert "unchanged" in final["error"]
        assert recorder.calls.count("MealPlanDraft") == graph.MAX_GENERATION_ATTEMPTS


class TestCritic:
    async def test_rejection_sends_the_plan_back_for_revision(
        self, wired, monkeypatch
    ):
        recorder = Recorder(
            critique=make_critique(
                approved=False,
                issues=["Day 4 pairs the heaviest session with the lightest day."],
            )
        )
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert recorder.calls.count("MealPlanDraft") == 2, "meals should be revised"
        assert final["saved_plan"] is not None, "a revised plan should still ship"

    async def test_critic_only_gets_one_round(self, wired, monkeypatch):
        """A critic allowed to keep asking would never stop."""
        recorder = Recorder(
            critique=make_critique(approved=False, issues=["Still not ideal."])
        )
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert recorder.calls.count("PlanCritique") == graph.MAX_CRITIQUE_ROUNDS
        assert final["saved_plan"] is not None

    async def test_critic_findings_reach_the_specialist(self, wired, monkeypatch):
        issue = "Day 4 has no rest despite three hard sessions before it."
        recorder = Recorder(critique=make_critique(approved=False, issues=[issue]))
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        await graph.run_agent("u1", today=TODAY)

        assert issue in recorder.meal_prompts[1], (
            "the revision prompt must name what the reviewer objected to"
        )

    async def test_a_failing_critic_does_not_cost_the_user_their_plan(
        self, wired, monkeypatch
    ):
        recorder = Recorder()

        def factory(schema):
            if schema is PlanCritique:
                class Broken:
                    async def ainvoke(self, _messages):
                        raise RuntimeError("critic exploded")

                return Broken()
            return recorder.factory(schema)

        monkeypatch.setattr(graph, "get_structured_llm", factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is not None
        assert "critique" in nodes_in(final)

    async def test_critic_approval_cannot_rescue_an_unsafe_plan(
        self, wired, monkeypatch
    ):
        """The deterministic validator outranks the model, always."""
        unsafe = make_meal_draft(TARGETS)
        unsafe.days[0].meals[0].name = "Grilled chicken salad"

        recorder = Recorder(
            meals=[unsafe],
            critique=make_critique(approved=True),  # critic waves it through
        )
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is None, (
            "an approving critic must not be able to bypass validation"
        )
        assert any("chicken" in e.lower() for e in final["validation_errors"])


class TestAssembly:
    async def test_missing_training_days_become_rest(self, wired, monkeypatch):
        """A short training draft shouldn't cost the user their meals."""
        partial = make_training_draft(days=3)
        recorder = Recorder(training=partial)
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)
        plan = final["saved_plan"]

        assert plan is not None
        assert len(plan.daily_plans) == 7
        assert plan.daily_plans[6].activity.duration_minutes == 0

        assemble = next(s for s in final["steps"] if s["node"] == "assemble")
        assert "set to rest" in assemble["message"]

    async def test_reasoning_combines_both_specialists(self, wired, monkeypatch):
        recorder = Recorder()
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)
        reasoning = final["saved_plan"].agent_reasoning

        assert "protein" in reasoning.lower()      # from the nutritionist
        assert "rest day" in reasoning.lower()     # from the trainer

    async def test_no_meals_fails_cleanly(self, wired, monkeypatch):
        def factory(schema):
            class Broken:
                async def ainvoke(self, _messages):
                    raise RuntimeError("model unavailable")

            return Broken()

        monkeypatch.setattr(graph, "get_structured_llm", factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["saved_plan"] is None
        assert final["error"] is not None


class TestNoActionPath:
    async def test_an_on_track_user_invokes_no_specialists(
        self, wired, monkeypatch
    ):
        """The cheapest run is the one that makes no calls at all."""
        plan = make_plan_in_db(TARGETS, reference_date=TODAY)

        class FakePlanRepo:
            @staticmethod
            async def get_active(_user_id):
                return plan

            @staticmethod
            async def save_new_version(p):
                raise AssertionError("should not save on a no-action run")

        monkeypatch.setattr(graph, "PlanRepository", FakePlanRepo)

        recorder = Recorder()
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)

        assert final["decision"] == AgentDecision.NO_ACTION
        assert recorder.calls == [], "no LLM should be called when nothing is wrong"
