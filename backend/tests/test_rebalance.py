"""What happens to a day you have already half-lived.

Skipping lunch with dinner still ahead is the scenario this whole product is
built around, and until now the execution did not match the claim.

Two faults with one cause. The prompt told the model to keep meals that had
already been eaten, while telling it only that one was eaten and two were
pending, never which. And because meal ids are deterministic, `d1-breakfast`
is `d1-breakfast` in every version of every plan, while the day's log matches
on id alone. So a replanned breakfast inherited the status of the one it
replaced: eat breakfast, skip lunch, press the button, and a completely
different breakfast came back marked "Eaten" with its calories counted toward
your day.
"""

from datetime import date

import pytest

from app.agent import graph
from app.agent.prompts import build_today_block
from app.models.enums import MealStatus
from app.models.log import DailyLogInDB, MealLogEntry
from tests.factories import make_plan_in_db, make_targets

TODAY = date(2026, 3, 15)
TARGETS = make_targets()


def a_log(**statuses) -> DailyLogInDB:
    """`a_log(breakfast="eaten", lunch="skipped")`."""
    return DailyLogInDB(
        user_id="u1",
        log_date=TODAY,
        meals=[
            MealLogEntry(meal_id=f"d1-{slot}", status=MealStatus(status))
            for slot, status in statuses.items()
        ],
    )


class TestThePromptKnowsWhatYouAte:
    """It was being asked to preserve something it could not see."""

    def _block(self, **statuses):
        plan = make_plan_in_db(TARGETS, duration_days=4)
        statuses_by_id = {
            f"d1-{slot}": status for slot, status in statuses.items()
        }
        return build_today_block(plan.daily_plans[0].meals, statuses_by_id)

    def test_an_eaten_meal_is_named_and_marked(self):
        block = self._block(breakfast="eaten")

        assert "breakfast" in block
        assert "ALREADY EATEN" in block

    def test_a_skipped_meal_is_marked_as_needing_absorbing(self):
        block = self._block(lunch="skipped")

        assert "SKIPPED" in block
        assert "absorbing" in block

    def test_unlogged_meals_are_marked_changeable(self):
        """Otherwise a rebalance has nothing it is allowed to touch."""
        block = self._block(breakfast="eaten")

        assert "STILL TO COME" in block

    def test_the_instruction_is_now_followable(self):
        """It names the meals rather than counting them."""
        block = self._block(breakfast="eaten", lunch="skipped")

        assert "same name and the same" in block
        assert "kcal" in block, "the model needs the numbers it must reproduce"

    def test_nothing_is_said_when_there_is_nothing_to_say(self):
        assert build_today_block([], {}) == ""


class TestEatenMealsSurviveTheReplan:
    """The code-level guarantee, which does not depend on the model obeying."""

    @staticmethod
    def _state(log):
        plan = make_plan_in_db(TARGETS, duration_days=4)
        return {"active_plan": plan, "today_log": log, "today": TODAY}, plan

    @staticmethod
    def _new_plan():
        """A freshly generated plan whose day 1 differs from the old one."""
        from tests.factories import make_health_plan

        plan = make_health_plan(TARGETS, days=4)
        for meal in plan.daily_plans[0].meals:
            meal.name = f"Something else for {meal.meal_type}"
            meal.calories_kcal += 111
        return plan

    def test_an_eaten_meal_comes_back_unchanged(self):
        state, old = self._state(a_log(breakfast="eaten"))
        new = self._new_plan()

        kept = graph._carry_over_what_already_happened(new, state)

        was = next(m for m in old.daily_plans[0].meals if m.meal_id == "d1-breakfast")
        now = next(
            m for m in new.daily_plans[0].meals if m.meal_id == "d1-breakfast"
        )

        assert kept == ["d1-breakfast"]
        assert now.name == was.name, (
            "the breakfast you actually ate was replaced by a different one, "
            "and your log still says you ate it"
        )
        assert now.calories_kcal == was.calories_kcal

    def test_a_skipped_meal_is_left_for_the_model_to_rewrite(self):
        """Rewriting the meals still ahead is the entire point of a rebalance."""
        state, _old = self._state(a_log(lunch="skipped"))
        new = self._new_plan()

        kept = graph._carry_over_what_already_happened(new, state)

        assert kept == []

    def test_a_substitution_counts_as_having_happened(self):
        state, _old = self._state(a_log(breakfast="substituted"))
        new = self._new_plan()

        assert graph._carry_over_what_already_happened(new, state) == ["d1-breakfast"]

    def test_only_today_is_protected(self):
        """Tomorrow has not happened yet and is fully the model's to change."""
        state, old = self._state(a_log(breakfast="eaten"))
        new = self._new_plan()
        for meal in new.daily_plans[1].meals:
            meal.name = "Tomorrow, rewritten"

        graph._carry_over_what_already_happened(new, state)

        assert new.daily_plans[1].meals[0].name == "Tomorrow, rewritten"

    def test_a_first_run_has_nothing_to_carry(self):
        new = self._new_plan()
        state = {"active_plan": None, "today_log": None, "today": TODAY}

        assert graph._carry_over_what_already_happened(new, state) == []

    def test_an_untouched_day_carries_nothing(self):
        state, _old = self._state(a_log())
        new = self._new_plan()

        assert graph._carry_over_what_already_happened(new, state) == []
