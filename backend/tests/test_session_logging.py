"""Skipping a workout has to reach the agent, like skipping a meal does.

For most of this project's life the loop had one input. Meals drove every
decision; training drove nothing. So a plan prescribing 45 minutes of lifting
to someone with 20 minutes and no gym was rewritten only when the food
happened to be wrong too, and a user who skipped every session was told
nothing needed changing.
"""

from datetime import date, timedelta

import pytest

from app.agent import graph
from app.models.enums import MealStatus, SessionStatus
from app.models.log import DailyLogInDB, MealLogEntry, SessionLogEntry
from app.services.adherence import build_snapshot
from tests.factories import make_plan_in_db, make_profile, make_targets

TODAY = date(2026, 3, 15)
TARGETS = make_targets()


def log_with_sessions(day: date, statuses, plan_day: int = 1) -> DailyLogInDB:
    return DailyLogInDB(
        user_id="u1",
        log_date=day,
        meals=[],
        sessions=[
            SessionLogEntry(plan_day=plan_day, status=status) for status in statuses
        ],
    )


class TestTheStreakIsCounted:
    def _snapshot(self, logs, today=TODAY):
        return build_snapshot(
            target_date=today,
            targets=TARGETS,
            plan=make_plan_in_db(TARGETS, duration_days=4),
            today_log=next((log for log in logs if log.log_date == today), None),
            recent_logs=logs,
        )

    def test_three_skipped_days_running_are_seen(self):
        logs = [
            log_with_sessions(TODAY - timedelta(days=n), [SessionStatus.SKIPPED])
            for n in range(3)
        ]

        assert self._snapshot(logs).session_skip_streak_days == 3

    def test_a_completed_session_breaks_the_streak(self):
        logs = [
            log_with_sessions(TODAY, [SessionStatus.SKIPPED]),
            log_with_sessions(TODAY - timedelta(days=1), [SessionStatus.DONE]),
            log_with_sessions(TODAY - timedelta(days=2), [SessionStatus.SKIPPED]),
        ]

        assert self._snapshot(logs).session_skip_streak_days == 1

    def test_a_day_with_nothing_logged_ends_the_streak(self):
        """Silence is not a skip.

        Someone who trains and forgets to tap the button has not told us they
        skipped, and counting it as one would let the agent rewrite a plan
        they are quietly following.
        """
        logs = [
            log_with_sessions(TODAY, [SessionStatus.SKIPPED]),
            DailyLogInDB(user_id="u1", log_date=TODAY - timedelta(days=1)),
            log_with_sessions(TODAY - timedelta(days=2), [SessionStatus.SKIPPED]),
        ]

        assert self._snapshot(logs).session_skip_streak_days == 1

    def test_todays_session_is_reported(self):
        logs = [log_with_sessions(TODAY, [SessionStatus.DONE])]
        snapshot = self._snapshot(logs)

        assert snapshot.session_done is True
        assert snapshot.session_skipped is False


class TestItChangesTheDecision:
    """The point of the whole exercise. A signal nothing reads is a checkbox."""

    @staticmethod
    def _decide(snapshot):
        plan = make_plan_in_db(TARGETS, duration_days=4, created_days_ago=1)
        state = {"today": TODAY, "force_replan": False}
        return graph._choose_action(state, snapshot, plan, TARGETS)

    def _snapshot_with_streak(self, days: int):
        logs = [
            log_with_sessions(TODAY - timedelta(days=n), [SessionStatus.SKIPPED])
            for n in range(days)
        ]
        return build_snapshot(
            target_date=TODAY,
            targets=TARGETS,
            plan=make_plan_in_db(TARGETS, duration_days=4, created_days_ago=1),
            today_log=logs[0] if logs else None,
            recent_logs=logs,
        )

    def test_three_skipped_sessions_trigger_a_replan(self):
        from app.models.enums import AgentDecision

        decision, detail = self._decide(self._snapshot_with_streak(3))

        assert decision == AgentDecision.STRUCTURAL_REPLAN
        assert "training" in detail.lower(), (
            "the rationale must say the training is what did not fit, not the "
            f"food: {detail!r}"
        )

    def test_two_skipped_sessions_are_left_alone(self):
        """A bad week is not a broken plan.

        Tearing up someone's training because they missed Monday and Tuesday
        is the behaviour that makes an app feel like it is nagging.
        """
        from app.models.enums import AgentDecision

        decision, _ = self._decide(self._snapshot_with_streak(2))

        assert decision != AgentDecision.STRUCTURAL_REPLAN

    def test_the_two_rationales_stay_distinguishable(self):
        """Folding food and training into one adherence number would produce a
        replan that explains nothing, and the explanation is the product."""
        import inspect

        source = inspect.getsource(graph._choose_action)

        assert "skipped meals" in source
        assert "skipped training" in source
