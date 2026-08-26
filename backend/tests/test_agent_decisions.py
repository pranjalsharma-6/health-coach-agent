"""Tests for the agent's decision rules and adherence computation.

The decision to intervene in someone's diet is made deterministically, which
means it can be tested exhaustively — no LLM, no database, no flakiness.
"""

from datetime import date, timedelta

from app.agent.graph import _choose_action
from app.models.enums import AgentDecision, MealStatus
from app.services.adherence import build_snapshot
from tests.factories import make_log, make_plan_in_db, make_targets

TODAY = date(2026, 3, 15)


def snapshot_for(plan, today_log=None, recent_logs=None, targets=None):
    targets = targets or make_targets()
    return build_snapshot(
        target_date=TODAY,
        targets=targets,
        plan=plan,
        today_log=today_log,
        recent_logs=recent_logs or ([today_log] if today_log else []),
    )


def state(today=TODAY, force_replan=False):
    return {"today": today, "force_replan": force_replan}


class TestDecisionRules:
    def test_no_plan_creates_initial(self):
        snapshot = snapshot_for(plan=None)
        decision, detail = _choose_action(state(), snapshot, None, make_targets())

        assert decision == AgentDecision.CREATE_INITIAL
        assert "first week" in detail.lower()

    def test_on_track_does_nothing(self):
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)
        # Plan created today, so day 1 applies.
        log = make_log(
            TODAY,
            [
                ("d1-breakfast", MealStatus.EATEN),
                ("d1-lunch", MealStatus.EATEN),
            ],
        )
        snapshot = snapshot_for(plan, log, [log], targets)

        decision, _ = _choose_action(state(), snapshot, plan, targets)
        assert decision == AgentDecision.NO_ACTION

    def test_skipped_meal_with_meals_remaining_rebalances_the_day(self):
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)
        log = make_log(
            TODAY,
            [
                ("d1-breakfast", MealStatus.EATEN),
                ("d1-lunch", MealStatus.SKIPPED),
            ],
        )
        snapshot = snapshot_for(plan, log, [log], targets)

        decision, detail = _choose_action(state(), snapshot, plan, targets)

        assert decision == AgentDecision.REBALANCE_DAY
        assert "skipped" in detail.lower()

    def test_skipped_meal_with_nothing_left_does_not_rebalance(self):
        """There's no point rebalancing a day that's already over."""
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)
        log = make_log(
            TODAY,
            [
                ("d1-breakfast", MealStatus.EATEN),
                ("d1-lunch", MealStatus.EATEN),
                ("d1-dinner", MealStatus.EATEN),
                ("d1-snack", MealStatus.SKIPPED),
            ],
        )
        snapshot = snapshot_for(plan, log, [log], targets)
        assert snapshot.meals_pending == 0

        decision, _ = _choose_action(state(), snapshot, plan, targets)
        assert decision == AgentDecision.NO_ACTION

    def test_three_day_skip_streak_triggers_structural_replan(self):
        """A pattern means the plan is wrong, not the user."""
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=3)

        recent = [
            make_log(TODAY - timedelta(days=n), [(f"d{n}-lunch", MealStatus.SKIPPED)])
            for n in range(3)
        ]
        snapshot = snapshot_for(plan, recent[0], recent, targets)

        assert snapshot.skip_streak_days >= 3
        decision, detail = _choose_action(state(), snapshot, plan, targets)

        assert decision == AgentDecision.STRUCTURAL_REPLAN
        assert "doesn't fit" in detail

    def test_low_seven_day_adherence_triggers_structural_replan(self):
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=2)

        # Mostly skips, but on non-consecutive days so the streak rule can't fire,
        # and with enough logged meals to clear the minimum-sample guard.
        recent = [
            make_log(
                TODAY - timedelta(days=1),
                [
                    ("d1-breakfast", MealStatus.SKIPPED),
                    ("d1-lunch", MealStatus.SKIPPED),
                    ("d1-dinner", MealStatus.SKIPPED),
                    ("d1-snack", MealStatus.EATEN),
                ],
            ),
            make_log(
                TODAY - timedelta(days=3),
                [
                    ("d2-breakfast", MealStatus.SKIPPED),
                    ("d2-lunch", MealStatus.SKIPPED),
                    ("d2-dinner", MealStatus.SKIPPED),
                    ("d2-snack", MealStatus.EATEN),
                ],
            ),
            make_log(
                TODAY - timedelta(days=5),
                [
                    ("d3-breakfast", MealStatus.SKIPPED),
                    ("d3-lunch", MealStatus.EATEN),
                ],
            ),
        ]
        today_log = make_log(TODAY, [])
        snapshot = snapshot_for(plan, today_log, recent + [today_log], targets)

        assert snapshot.meals_logged_7d >= 8
        assert snapshot.adherence_rate_7d < 0.6
        decision, _ = _choose_action(state(), snapshot, plan, targets)
        assert decision == AgentDecision.STRUCTURAL_REPLAN

    def test_low_adherence_on_a_tiny_sample_does_not_trigger_a_replan(self):
        """One skip out of two logged meals is 50% adherence but no evidence.

        Without a minimum-sample guard the agent tears up a brand-new plan on
        the user's first missed breakfast, which is exactly the overreaction
        the structural rule exists to avoid.
        """
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)
        log = make_log(
            TODAY,
            [
                ("d1-breakfast", MealStatus.EATEN),
                ("d1-lunch", MealStatus.SKIPPED),
            ],
        )
        snapshot = snapshot_for(plan, log, [log], targets)

        assert snapshot.adherence_rate_7d < 0.6
        assert snapshot.meals_logged_7d < 8

        decision, _ = _choose_action(state(), snapshot, plan, targets)

        # Rebalances the day rather than restructuring the whole plan.
        assert decision == AgentDecision.REBALANCE_DAY

    def test_expired_plan_triggers_the_next_block(self):
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=8, duration_days=7)
        snapshot = snapshot_for(plan, make_log(TODAY, []), targets=targets)

        decision, detail = _choose_action(state(), snapshot, plan, targets)

        assert decision == AgentDecision.STRUCTURAL_REPLAN
        assert "complete" in detail.lower()

    def test_calorie_overage_with_meals_left_rebalances(self):
        targets = make_targets(calories=2000, protein=170)
        plan = make_plan_in_db(targets, reference_date=TODAY)
        log = make_log(
            TODAY,
            [
                ("d1-breakfast", MealStatus.EATEN),
                ("d1-lunch", MealStatus.EATEN),
            ],
        )
        snapshot = snapshot_for(plan, log, [log], targets)
        # Force an overage without touching the skip path.
        snapshot.calories_consumed = 2600
        snapshot.calories_remaining = -600

        decision, detail = _choose_action(state(), snapshot, plan, targets)

        assert decision == AgentDecision.REBALANCE_DAY
        assert "2600" in detail

    def test_force_replan_overrides_being_on_track(self):
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)
        snapshot = snapshot_for(plan, make_log(TODAY, []), targets=targets)

        decision, _ = _choose_action(
            state(force_replan=True), snapshot, plan, targets
        )
        assert decision == AgentDecision.STRUCTURAL_REPLAN

    def test_severity_ordering_streak_beats_single_skip(self):
        """A streak must win over the day-rebalance rule, not the other way round."""
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=3)

        # Created 3 days ago, so today is day 4 of the plan and today's log has
        # to reference that day's meal ids for the skip to be counted.
        recent = [
            make_log(TODAY, [("d4-lunch", MealStatus.SKIPPED)]),
            make_log(TODAY - timedelta(days=1), [("d3-lunch", MealStatus.SKIPPED)]),
            make_log(TODAY - timedelta(days=2), [("d2-lunch", MealStatus.SKIPPED)]),
        ]
        snapshot = snapshot_for(plan, recent[0], recent, targets)

        # Both conditions hold simultaneously.
        assert snapshot.meals_skipped > 0
        assert snapshot.skip_streak_days >= 3

        decision, _ = _choose_action(state(), snapshot, plan, targets)
        assert decision == AgentDecision.STRUCTURAL_REPLAN


class TestAdherenceComputation:
    def test_unlogged_days_are_not_counted_as_failures(self):
        """Silence is missing data, not non-adherence.

        Otherwise the agent panics every time someone goes on holiday.
        """
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)
        snapshot = snapshot_for(plan, make_log(TODAY, []), [], targets)

        assert snapshot.adherence_rate_7d == 1.0
        assert snapshot.meals_skipped == 0

    def test_consumed_totals_use_planned_macros_when_actuals_absent(self):
        targets = make_targets(calories=2000, protein=170)
        plan = make_plan_in_db(targets, reference_date=TODAY)
        log = make_log(TODAY, [("d1-breakfast", MealStatus.EATEN)])

        snapshot = snapshot_for(plan, log, [log], targets)
        breakfast = plan.daily_plans[0].meals[0]

        assert snapshot.calories_consumed == breakfast.calories_kcal
        assert snapshot.protein_consumed_g == breakfast.protein_g

    def test_substituted_meal_counts_as_eaten_with_actual_macros(self):
        from app.models.log import MealLogEntry

        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY)
        log = make_log(TODAY, [])
        log.meals = [
            MealLogEntry(
                meal_id="d1-breakfast",
                status=MealStatus.SUBSTITUTED,
                actual_calories_kcal=320,
                actual_protein_g=18,
                substitute_name="Two idlis and sambar",
            )
        ]

        snapshot = snapshot_for(plan, log, [log], targets)

        assert snapshot.meals_eaten == 1
        assert snapshot.calories_consumed == 320
        assert snapshot.protein_consumed_g == 18

    def test_skip_streak_breaks_on_a_compliant_day(self):
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=4)

        recent = [
            make_log(TODAY, [("d1-lunch", MealStatus.SKIPPED)]),
            make_log(TODAY - timedelta(days=1), [("d2-lunch", MealStatus.EATEN)]),
            make_log(TODAY - timedelta(days=2), [("d3-lunch", MealStatus.SKIPPED)]),
        ]
        snapshot = snapshot_for(plan, recent[0], recent, targets)

        assert snapshot.skip_streak_days == 1

    def test_plan_day_wraps_when_user_runs_past_the_plan(self):
        """Day 8 of a 7-day plan maps back to day 1 rather than crashing."""
        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=TODAY, created_days_ago=7, duration_days=7)
        log = make_log(TODAY, [("d1-breakfast", MealStatus.EATEN)])

        snapshot = snapshot_for(plan, log, [log], targets)

        assert snapshot.meals_planned == 4
        assert snapshot.meals_eaten == 1
