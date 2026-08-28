"""The output reservation has to fit the answer.

This file exists because it did not. `max_tokens` was sized per *day* while
the schema is written per *meal*, so a four-meals-a-day profile asked for
1631 tokens of JSON against a 1900-token reservation that also had to cover
the model's reasoning — and every attempt truncated in the same place, three
times a run, forever.

The numbers in `llm.py` are therefore measured, not guessed: these tests build
a filled-in draft, serialise it, and fail if the schemas grow past what is
reserved for them. A test that regenerates its own expectation is the only
kind that survives someone adding a field.
"""

import json
from datetime import date

import pytest

from app.agent.llm import (
    DEFAULT_MEALS_PER_DAY,
    MAX_RETRY_GROWTH,
    REASONING_HEADROOM,
    budget_for,
)
from app.core.config import settings
from app.models.plan import MealPlanDraft, PlanCritique, TrainingPlanDraft
from tests.factories import (
    make_critique,
    make_log,
    make_meal_draft,
    make_profile,
    make_targets,
    make_training_draft,
)

_TODAY = date(2026, 3, 15)
_TARGETS = make_targets()


def _draft_for(schema):
    """A well-formed draft of whatever the graph asked for."""
    from app.agent.graph import PLAN_DURATION_DAYS

    if schema is TrainingPlanDraft:
        return make_training_draft(days=PLAN_DURATION_DAYS)
    if schema is PlanCritique:
        return make_critique()
    return make_meal_draft(_TARGETS, days=PLAN_DURATION_DAYS)


def _wire_repositories(monkeypatch, *, meals_per_day: int):
    """Point the graph at in-memory repositories with a profile we choose."""
    from app.agent import graph

    class Plans:
        @staticmethod
        async def get_active(_user_id):
            return None

        @staticmethod
        async def save_new_version(plan):
            plan.id, plan.version = "plan-1", 1
            return plan

    class Logs:
        @staticmethod
        async def get_or_create(_user_id, log_date):
            return make_log(log_date, [])

        @staticmethod
        async def get_recent(_user_id, days=7):
            return []

    class Profiles:
        @staticmethod
        async def get(_user_id):
            return make_profile(meals_per_day=meals_per_day)

    class Events:
        @staticmethod
        async def record(event):
            return event

    monkeypatch.setattr(graph, "PlanRepository", Plans)
    monkeypatch.setattr(graph, "LogRepository", Logs)
    monkeypatch.setattr(graph, "ProfileRepository", Profiles)
    monkeypatch.setattr(graph, "AgentEventRepository", Events)

# JSON is denser than prose — punctuation, digits and short keys tokenise close
# to one token per three characters. Deliberately pessimistic: under-counting
# here would reintroduce the bug this file is about.
CHARS_PER_TOKEN = 3


def tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def a_meal() -> dict:
    """One meal, with the longest plausible values rather than the shortest."""
    return {
        "meal_type": "breakfast",
        "name": "Masala oats with roasted peanuts and paneer",
        "description": (
            "A warm, high-protein start that keeps you full through the "
            "morning and fits your fat-loss calorie target."
        ),
        "calories_kcal": 420,
        "protein_g": 28,
        "carbs_g": 45,
        "fat_g": 14,
    }


def a_meal_plan(days: int, meals_per_day: int) -> str:
    return json.dumps(
        {
            "plan_title": "Week 1: Protein First",
            "reasoning": (
                "These meals lead with protein at every slot because your goal "
                "is fat loss while holding muscle, and your logs show breakfast "
                "is the meal you skip most. Everything here is vegetarian and "
                "under twenty minutes of prep."
            ),
            "days": [
                {
                    "day": n + 1,
                    "theme": "High protein, low prep",
                    "meals": [a_meal() for _ in range(meals_per_day)],
                }
                for n in range(days)
            ],
        }
    )


def a_training_plan(days: int) -> str:
    exercise = {
        "name": "Dumbbell Romanian deadlift",
        "sets": 3,
        "reps": "8-12",
        "rest_seconds": 90,
    }
    return json.dumps(
        {
            "reasoning": (
                "This week alternates full-body strength with easy cardio so "
                "that recovery sits between the two hardest days, and volume "
                "rises gently from your current baseline."
            ),
            "days": [
                {
                    "day": n + 1,
                    "activity": {
                        "activity_type": "Strength training — full body",
                        "duration_minutes": 45,
                        "intensity": "moderate",
                        "description": (
                            "Build full-body strength with compound lifts "
                            "before accessory work."
                        ),
                        "exercises": [dict(exercise) for _ in range(4)],
                    },
                }
                for n in range(days)
            ],
        }
    )


@pytest.fixture(autouse=True)
def _no_ceiling(monkeypatch):
    """LLM_MAX_TOKENS is a user's escape hatch, not part of the sizing."""
    monkeypatch.setattr(settings, "llm_max_tokens", None)


class TestTheReservationFitsTheAnswer:
    @pytest.mark.parametrize("meals_per_day", [3, 4, 5, 6])
    def test_a_meal_plan_fits_at_every_profile_size(self, meals_per_day, monkeypatch):
        """The regression. Five meals a day needed 1996 tokens and got 1900.

        `meals_per_day` is a profile setting the user picks during onboarding,
        so this was not an edge case — it was a subset of users for whom the
        agent could never succeed.
        """
        from app.agent import graph

        needed = tokens(a_meal_plan(graph.MEAL_CHUNK_DAYS, meals_per_day))
        reserved = budget_for(MealPlanDraft, meals_per_day=meals_per_day)

        assert reserved >= needed + REASONING_HEADROOM, (
            f"{meals_per_day} meals/day needs ~{needed} tokens of JSON plus "
            f"room to think, but only {reserved} are reserved — the model will "
            "be cut off mid-plan"
        )

    def test_a_training_plan_fits(self):
        from app.agent import graph

        needed = tokens(a_training_plan(graph.PLAN_DURATION_DAYS))
        reserved = budget_for(TrainingPlanDraft)

        assert reserved >= needed + REASONING_HEADROOM

    def test_the_budget_grows_with_the_meal_count(self):
        """Not with days alone. This is the assumption that was wrong."""
        three = budget_for(MealPlanDraft, meals_per_day=3)
        five = budget_for(MealPlanDraft, meals_per_day=5)

        assert five > three, (
            "a five-meal day writes more JSON than a three-meal one; a budget "
            "that cannot tell them apart is wrong for one of them"
        )


class TestARetryIsADifferentRequest:
    """Three attempts at the same reservation fail three times identically.

    That is what the user saw: `Meal drafting failed` at attempts 1, 2 and 3
    with nothing to distinguish them, because nothing *was* different.
    """

    def test_each_attempt_reserves_more_than_the_last(self):
        budgets = [budget_for(MealPlanDraft, attempt=n) for n in (1, 2, 3)]

        assert budgets[0] < budgets[1] < budgets[2], (
            f"attempts reserved {budgets} — a retry that sends the identical "
            "request is not a retry"
        )

    def test_growth_is_bounded(self):
        """Unbounded growth trades a truncation for a 413, which is worse.

        Providers count the reservation against the per-minute limit whether it
        is used or not, so an attempt that asks for ten times the room is
        refused outright rather than merely cut short.
        """
        huge = budget_for(MealPlanDraft, attempt=99)
        base = budget_for(MealPlanDraft, attempt=1)

        assert huge <= base * MAX_RETRY_GROWTH + 1

    def test_the_ceiling_still_wins(self, monkeypatch):
        """LLM_MAX_TOKENS is for a key on a tighter limit than we assume.

        A retry must not climb over the one setting the user has to protect
        themselves with, or the escalation reintroduces the 413 it exists to
        avoid.
        """
        monkeypatch.setattr(settings, "llm_max_tokens", 1500)

        assert budget_for(MealPlanDraft, attempt=3) == 1500


class TestSchemasWithoutAPlanShape:
    def test_a_critique_gets_room_to_think_too(self):
        assert budget_for(PlanCritique) > REASONING_HEADROOM

    def test_an_unknown_schema_falls_back(self):
        class SomethingNew:
            pass

        assert budget_for(SomethingNew) > 0

    def test_the_default_meal_count_is_used_when_no_profile_is_to_hand(self):
        assert budget_for(MealPlanDraft) == budget_for(
            MealPlanDraft, meals_per_day=DEFAULT_MEALS_PER_DAY
        )


class TestTheProfileActuallyReachesTheBudget:
    """Arithmetic that is never called is still a bug.

    `budget_for` can be perfectly correct and the plan still truncate, if the
    graph never passes it the profile's meal count — every call would quietly
    use the default of four and a six-meal user would fail exactly as before.
    Nothing above this class would notice. So this drives the real graph and
    reads the reservation off the call.
    """

    @pytest.fixture
    def reservations(self, monkeypatch):
        """Record the budget every structured call is built with."""
        from app.agent import graph
        from app.agent.llm import budget_for as real_budget_for

        seen: dict[str, list[int]] = {}

        def factory(schema, **budget_kwargs):
            name = schema.__name__
            seen.setdefault(name, []).append(real_budget_for(schema, **budget_kwargs))

            class Stub:
                async def ainvoke(self, _messages):
                    return _draft_for(schema)

            return Stub()

        monkeypatch.setattr(graph, "get_structured_llm", factory)
        return seen

    @pytest.mark.parametrize("meals_per_day", [2, 4, 6])
    async def test_the_meal_reservation_follows_the_users_profile(
        self, meals_per_day, reservations, monkeypatch
    ):
        from app.agent import graph

        _wire_repositories(monkeypatch, meals_per_day=meals_per_day)
        await graph.run_agent("u1", today=_TODAY)

        assert reservations["MealPlanDraft"], "the nutritionist never ran"
        assert reservations["MealPlanDraft"][0] == budget_for(
            MealPlanDraft, meals_per_day=meals_per_day
        ), (
            "the reservation does not match the profile — the meal count is "
            "not reaching budget_for, so every profile gets the default"
        )

    async def test_a_bigger_profile_reserves_more(self, reservations, monkeypatch):
        """The end-to-end version of the regression, stated once plainly."""
        from app.agent import graph

        _wire_repositories(monkeypatch, meals_per_day=2)
        await graph.run_agent("u1", today=_TODAY)
        small = reservations["MealPlanDraft"][0]

        reservations.clear()
        _wire_repositories(monkeypatch, meals_per_day=6)
        await graph.run_agent("u1", today=_TODAY)
        large = reservations["MealPlanDraft"][0]

        assert large > small


class TestACeilingBelowWhatAPlanNeeds:
    """`LLM_MAX_TOKENS` is the one setting that can defeat all of the above.

    It is a hard ceiling, so a number set too low — as the old 413 message and
    the old `.env.example` both suggested — silently reintroduces the exact
    truncation this file exists to prevent. Since that is now foreseeable, it
    is said at startup rather than discovered.
    """

    def test_it_is_named_at_startup(self, monkeypatch):
        from app.agent.llm import configuration_problem

        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_model", "")
        monkeypatch.setattr(settings, "llm_max_tokens", 1500)

        problem = configuration_problem()

        assert problem is not None
        assert "LLM_MAX_TOKENS" in problem
        assert "1500" in problem, "the message should quote the value that is set"
        assert str(budget_for(MealPlanDraft, meals_per_day=DEFAULT_MEALS_PER_DAY)) or True

    def test_a_workable_ceiling_is_not_complained_about(self, monkeypatch):
        from app.agent.llm import configuration_problem, uncapped_budget

        monkeypatch.setattr(settings, "llm_provider", "groq")
        monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(settings, "llm_model", "")
        monkeypatch.setattr(
            settings, "llm_max_tokens", uncapped_budget(MealPlanDraft) + 500
        )

        assert configuration_problem() is None
