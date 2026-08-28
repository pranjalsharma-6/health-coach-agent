"""Tests for the multi-agent planning pipeline.

These check the claims the architecture actually makes — that the specialists
run concurrently, that a retry redraws only what needs redrawing, and that the
critic can never make an unsafe plan safe — rather than just that the graph
compiles.
"""

import asyncio
import re
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

    @staticmethod
    def _requested_days(text: str) -> tuple[int, int] | None:
        """The day range this call asked for, if it is a chunked request."""
        match = re.search(r"DAYS (\d+) TO (\d+) ONLY", text)
        return (int(match.group(1)), int(match.group(2))) if match else None

    def factory(self, schema, **_budget):
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
                draft = (
                    recorder.meals.pop(0)
                    if len(recorder.meals) > 1
                    else recorder.meals[0]
                )

                # A real model returns only the days it was asked for. Returning
                # the whole week per chunk would stitch into a fourteen-day plan
                # and hide whatever the test is actually checking.
                window = recorder._requested_days(text)
                if window is None:
                    return draft

                first, last = window
                return draft.model_copy(
                    update={
                        "days": [d for d in draft.days if first <= d.day <= last]
                    }
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
        assert recorder.calls.count("MealPlanDraft") == 2 * graph.meal_chunk_count(), (
            "meals should be redrawn"
        )
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
        assert recorder.calls.count("MealPlanDraft") == (
            graph.MAX_GENERATION_ATTEMPTS * graph.meal_chunk_count()
        )


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

        assert recorder.calls.count("MealPlanDraft") == 2 * graph.meal_chunk_count(), (
            "meals should be revised"
        )
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

        assert any(issue in p for p in recorder.meal_prompts), (
            "the revision prompt must name what the reviewer objected to"
        )

    async def test_a_failing_critic_does_not_cost_the_user_their_plan(
        self, wired, monkeypatch
    ):
        recorder = Recorder()

        def factory(schema, **_budget):
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
        # One day short of the plan, whatever the plan's length is.
        short_by_one = graph.PLAN_DURATION_DAYS - 1
        recorder = Recorder(training=make_training_draft(days=short_by_one))
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)
        plan = final["saved_plan"]

        assert plan is not None
        assert len(plan.daily_plans) == graph.PLAN_DURATION_DAYS
        assert plan.daily_plans[-1].activity.duration_minutes == 0

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
        def factory(schema, **_budget):
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


class TestChunkedMealDrafting:
    """The week is drafted in day ranges rather than all at once.

    Seven days of four meals is 28 nested objects, and models lose count over
    that distance. The observed run produced, in order: a two-day week, a day
    with three meals, and an egg in a vegetarian plan — consistency failures
    across a long output, not errors of judgement. The codebase already argues
    that small structured outputs are the reliable ones; this applies it one
    level down from the specialist split.
    """

    def test_the_week_is_covered_exactly_once(self):
        ranges = graph._chunk_ranges(graph.PLAN_DURATION_DAYS, graph.MEAL_CHUNK_DAYS)
        covered = [day for first, last in ranges for day in range(first, last + 1)]

        assert covered == list(range(1, graph.PLAN_DURATION_DAYS + 1)), (
            "chunks must tile the week with no gap and no overlap"
        )

    def test_a_long_plan_is_split(self):
        """Chunking is what a seven-day plan needs; a four-day one fits in a
        single call, so assert the behaviour rather than the current config."""
        assert len(graph._chunk_ranges(7, graph.MEAL_CHUNK_DAYS)) > 1

    def test_the_chunk_count_matches_the_configured_length(self):
        assert graph.meal_chunk_count() == len(
            graph._chunk_ranges(graph.PLAN_DURATION_DAYS, graph.MEAL_CHUNK_DAYS)
        )

    def test_no_chunk_is_longer_than_the_limit(self):
        for first, last in graph._chunk_ranges(
            graph.PLAN_DURATION_DAYS, graph.MEAL_CHUNK_DAYS
        ):
            assert last - first + 1 <= graph.MEAL_CHUNK_DAYS

    @pytest.mark.parametrize(
        "total,size,expected",
        [
            (7, 4, [(1, 4), (5, 7)]),
            (7, 3, [(1, 3), (4, 6), (7, 7)]),
            (4, 4, [(1, 4)]),
            (1, 4, [(1, 1)]),
        ],
    )
    def test_range_arithmetic(self, total, size, expected):
        assert graph._chunk_ranges(total, size) == expected

    async def test_each_chunk_is_told_which_days_to_return(self, wired, monkeypatch):
        recorder = Recorder()
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        await graph.run_agent("u1", today=TODAY)

        windows = [recorder._requested_days(p) for p in recorder.meal_prompts]
        assert all(w is not None for w in windows), (
            "every meal call must name its day range"
        )
        assert sorted(windows) == graph._chunk_ranges(
            graph.PLAN_DURATION_DAYS, graph.MEAL_CHUNK_DAYS
        )

    async def test_the_chunks_stitch_into_one_full_week(self, wired, monkeypatch):
        recorder = Recorder()
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        final = await graph.run_agent("u1", today=TODAY)
        plan = final["saved_plan"]

        assert plan is not None
        assert [d.day for d in plan.daily_plans] == list(
            range(1, graph.PLAN_DURATION_DAYS + 1)
        )

    @pytest.mark.skipif(
        "graph.meal_chunk_count() < 2",
        reason="only meaningful when the plan is long enough to split",
    )
    async def test_the_chunks_run_concurrently(self, wired, monkeypatch):
        """Sequential chunks would double the wait for the same week."""
        recorder = Recorder(delay=0.15)
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        await graph.run_agent("u1", today=TODAY)

        meal_starts = [t for name, t in recorder.started if name == "MealPlanDraft"]
        meal_ends = [t for name, t in recorder.finished if name == "MealPlanDraft"]

        assert len(meal_starts) == graph.meal_chunk_count()
        assert max(meal_starts) < min(meal_ends), (
            "the second chunk started only after the first finished"
        )


class TestFeedbackCannotBreakTheDiet:
    """Constraints are restated after any feedback.

    The reviewer asked for more variety and the next draft answered with eggs
    in a vegetarian plan. The diet rules were in the prompt — they had just
    stopped being the last thing the model read.
    """

    async def test_the_non_negotiables_follow_the_critique(
        self, wired, monkeypatch
    ):
        recorder = Recorder(
            critique=make_critique(
                approved=False, issues=["The same meals repeat every day."]
            )
        )
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        await graph.run_agent("u1", today=TODAY)

        revision = next(
            p for p in recorder.meal_prompts if "repeat every day" in p
        )
        assert "THESE OVERRIDE EVERYTHING ABOVE" in revision
        assert revision.index("repeat every day") < revision.index(
            "THESE OVERRIDE EVERYTHING ABOVE"
        ), "the constraints must come after the feedback, not before it"

    async def test_a_first_attempt_has_no_such_section(self, wired, monkeypatch):
        """Nothing to override when there is no feedback."""
        recorder = Recorder()
        monkeypatch.setattr(graph, "get_structured_llm", recorder.factory)

        await graph.run_agent("u1", today=TODAY)

        assert "THESE OVERRIDE EVERYTHING ABOVE" not in recorder.meal_prompts[0]
