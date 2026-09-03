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

    def _block(self, targets=False, **statuses):
        plan = make_plan_in_db(TARGETS, duration_days=4)
        statuses_by_id = {
            f"d1-{slot}": status for slot, status in statuses.items()
        }
        return build_today_block(
            plan.daily_plans[0].meals,
            statuses_by_id,
            TARGETS if targets else None,
        )

    def test_an_eaten_meal_is_named_and_marked(self):
        block = self._block(breakfast="eaten")

        assert "breakfast" in block
        assert "ALREADY EATEN" in block

    def test_a_skipped_meal_is_marked_as_gone_rather_than_pending(self):
        """It has happened. It is not a meal still ahead of the user."""
        block = self._block(lunch="skipped")

        assert "SKIPPED AND GONE" in block
        assert "does not count toward today" in block

    def test_the_gap_is_stated_as_a_number_per_meal(self):
        """A number the model can act on beats an instruction to redistribute.

        And it has to say it overrides the per-meal average from the targets
        block, which is still telling the same model that meals average a
        quarter of the day. Given two instructions it cannot both satisfy, the
        model split the difference and every attempt failed validation.
        """
        block = self._block(breakfast="eaten", lunch="skipped", targets=True)

        assert "kcal each" in block
        assert "overrides the per-meal average" in block

    def test_the_day_is_stated_as_a_smaller_day(self):
        """Skipping a meal means eating less that day. Saying otherwise asks
        two remaining meals to carry a whole missed lunch."""
        block = self._block(breakfast="eaten", lunch="skipped", targets=True)

        assert "not a" in block and "kcal day" in block
        assert "some of a missed meal is made up and some is simply gone" in block

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

    def test_a_skipped_meal_is_also_kept(self):
        """It stays in the plan, and stops counting toward the day.

        The first version of this rewrote skipped meals, on the reasoning that
        they were still ahead of the user. They are not: a skipped lunch is as
        settled as an eaten breakfast. Regenerating it at full size is what
        left no room for dinner to grow into, so a rebalance produced a plan
        that looked identical to the one it replaced.
        """
        state, old = self._state(a_log(lunch="skipped"))
        new = self._new_plan()

        kept = graph._carry_over_what_already_happened(new, state)

        was = next(m for m in old.daily_plans[0].meals if m.meal_id == "d1-lunch")
        now = next(m for m in new.daily_plans[0].meals if m.meal_id == "d1-lunch")

        assert kept == ["d1-lunch"]
        assert now.name == was.name

    def test_the_pending_meals_are_still_the_models_to_rewrite(self):
        """Only what has happened is fixed. The rest of the day is the point."""
        state, _old = self._state(a_log(breakfast="eaten", lunch="skipped"))
        new = self._new_plan()

        graph._carry_over_what_already_happened(new, state)

        untouched = [
            meal.name
            for meal in new.daily_plans[0].meals
            if meal.meal_id in ("d1-dinner", "d1-snack")
        ]
        assert all("Something else" in name for name in untouched), (
            "dinner and snack were overwritten, so nothing can absorb the skip"
        )

    def test_half_a_skipped_meal_is_absorbed_and_half_is_forfeited(self):
        """The arithmetic that makes a real rebalance both possible and sane.

        While a skipped meal still counted in full, the day was already at
        target and there was nowhere for dinner to grow into. Excluding it
        entirely went too far the other way: two remaining meals had to carry a
        whole missed lunch, which meant an 873 kcal snack that no model would
        write and no person would eat, so every attempt failed and the run gave
        up after three.
        """
        from app.agent.validators import (
            SKIPPED_MEAL_ABSORPTION,
            validate_plan,
        )
        from tests.factories import make_health_plan, make_profile

        targets = make_targets()
        profile = make_profile()

        def day_one_absorbing(fraction: float):
            plan = make_health_plan(targets, days=4)
            day = plan.daily_plans[0]
            lunch = next(m for m in day.meals if m.meal_id == "d1-lunch")
            dinner = next(m for m in day.meals if m.meal_id == "d1-dinner")
            dinner.calories_kcal += round(lunch.calories_kcal * fraction)
            dinner.protein_g += round(lunch.protein_g * fraction)
            dinner.carbs_g += round(lunch.carbs_g * fraction)
            dinner.fat_g += round(lunch.fat_g * fraction)
            return plan

        def kcal_errors(plan):
            result = validate_plan(
                plan, profile, targets, skipped_today={"d1-lunch"}
            )
            return [e for e in result.errors if "Day 1" in e and "kcal" in e]

        assert not kcal_errors(day_one_absorbing(SKIPPED_MEAL_ABSORPTION)), (
            "absorbing half a skipped meal is the target behaviour and must pass"
        )
        assert kcal_errors(day_one_absorbing(0.0)), (
            "absorbing none of it leaves the day short, which is the complaint "
            "that started this"
        )
        assert kcal_errors(day_one_absorbing(1.0)), (
            "absorbing all of it is the over-correction that made every attempt "
            "fail"
        )
