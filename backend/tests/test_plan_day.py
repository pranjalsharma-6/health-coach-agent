"""Which day of the plan "today" maps to.

There used to be two implementations: a calendar-date difference on the server
and elapsed-milliseconds-over-24-hours on the client. They disagree for the
window after midnight but before the 24-hour mark, and the disagreement is
invisible — the client renders one day's meals while the server computes
adherence against another. Since the two are matched by meal_id, nothing lines
up: every logged meal reads as unlogged, and the header sits at "0 of 4 eaten"
while each card shows "Eaten".

The server now publishes the resolved day and the client uses it.
"""

from datetime import date, datetime, time

class TestPlanDayIsPublished:
    """The client used to re-derive which plan day "today" is, and got it wrong.

    Adherence is computed against one day's meals, matched by meal_id. If the
    client renders a different day, none of the ids the server counted appear
    on screen — the header reads "0 of 4 eaten" while every card shows "Eaten".
    Publishing the resolved day removes the second implementation.
    """

    @staticmethod
    def _snapshot(created_at, target):
        from app.models.log import DailyLogInDB
        from app.services.adherence import build_snapshot
        from tests.factories import make_plan_in_db, make_targets

        targets = make_targets()
        plan = make_plan_in_db(targets, reference_date=created_at)
        plan.created_at = datetime.combine(created_at, time(23, 0))

        return build_snapshot(
            target_date=target,
            targets=targets,
            plan=plan,
            today_log=DailyLogInDB(user_id="u", log_date=target, meals=[]),
            recent_logs=[],
        )

    def test_the_day_is_reported(self):
        created = date(2026, 3, 15)
        assert self._snapshot(created, created).plan_day == 1

    def test_it_advances_with_the_calendar_not_the_clock(self):
        """A plan created at 23:00 is on day 2 at 01:00 the next morning — two
        hours later. Dividing elapsed milliseconds by 24 hours says day 1, and
        that disagreement is the bug."""
        created = date(2026, 3, 15)
        assert self._snapshot(created, date(2026, 3, 16)).plan_day == 2

    def test_it_wraps_past_the_end_of_the_plan(self):
        created = date(2026, 3, 15)
        # Day 8 of a 7-day plan is day 1 again.
        assert self._snapshot(created, date(2026, 3, 22)).plan_day == 1

    def test_it_is_none_without_a_plan(self):
        from app.models.log import DailyLogInDB
        from app.services.adherence import build_snapshot
        from tests.factories import make_targets

        target = date(2026, 3, 15)
        snapshot = build_snapshot(
            target_date=target,
            targets=make_targets(),
            plan=None,
            today_log=DailyLogInDB(user_id="u", log_date=target, meals=[]),
            recent_logs=[],
        )
        assert snapshot.plan_day is None
