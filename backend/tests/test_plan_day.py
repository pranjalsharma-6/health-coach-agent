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

import pytest
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


class TestPlanLength:
    """A short week used to pass every structural check.

    The numbering rule asked whether day numbers were sequential, and [1, 2]
    is. So a run that reported "Drafted 2 days of meals" would have shipped a
    two-day plan as a seven-day one, and the dashboard would have wrapped
    around to day 1 on Wednesday.
    """

    @staticmethod
    def _plan_and_profile(days):
        from app.models.enums import DietType
        from tests.factories import make_health_plan, make_profile, make_targets

        targets = make_targets()
        return (
            make_health_plan(targets, days=days),
            make_profile(diet_type=DietType.VEGETARIAN),
            targets,
        )

    def test_a_plan_too_short_to_use_is_rejected(self):
        from app.agent.validators import validate_plan

        plan, profile, targets = self._plan_and_profile(2)
        result = validate_plan(plan, profile, targets, expected_days=7)

        assert not result.is_valid
        assert any("too few to be usable" in e for e in result.errors)

    def test_a_slightly_short_plan_ships_with_a_warning(self):
        """Models come up a day short on this kind of output, and a correct
        five-day plan beats a fourth attempt that also fails. It rotates
        sooner, which is true and worth saying, not a reason to bin it."""
        from app.agent.validators import validate_plan

        plan, profile, targets = self._plan_and_profile(5)
        result = validate_plan(plan, profile, targets, expected_days=7)

        assert result.is_valid
        assert any("rotate sooner" in w for w in result.warnings)

    def test_the_shortfall_is_never_silent(self):
        from app.agent.validators import validate_plan

        plan, profile, targets = self._plan_and_profile(3)
        result = validate_plan(plan, profile, targets, expected_days=4)

        assert result.is_valid
        assert result.warnings, "a short plan must say so"

    def test_a_plan_longer_than_requested_is_rejected(self):
        """Tolerance runs one way only. Extra days were not asked for and cost
        the user time they did not agree to spend."""
        from app.agent.validators import validate_plan

        plan, profile, targets = self._plan_and_profile(9)
        result = validate_plan(plan, profile, targets, expected_days=7)

        assert not result.is_valid
        assert any("more than" in e for e in result.errors)

    def test_the_right_length_passes(self):
        from app.agent.validators import validate_plan

        plan, profile, targets = self._plan_and_profile(7)
        assert validate_plan(plan, profile, targets, expected_days=7).is_valid

    def test_the_check_is_opt_in(self):
        """Callers that do not know the intended length skip it rather than
        guessing at one."""
        from app.agent.validators import validate_plan

        plan, profile, targets = self._plan_and_profile(2)
        assert validate_plan(plan, profile, targets).is_valid

    def test_the_floor_holds_even_for_a_short_request(self):
        """70% of 3 is 2.1, but two days on repeat is not a plan."""
        from app.agent.validators import MIN_USABLE_PLAN_DAYS, validate_plan

        plan, profile, targets = self._plan_and_profile(2)
        assert not validate_plan(plan, profile, targets, expected_days=3).is_valid
        assert MIN_USABLE_PLAN_DAYS == 3

    @pytest.mark.parametrize("days", [1, 2])
    def test_the_floor_never_exceeds_the_request(self, days):
        """Asking for one day and getting one day is not a shortfall. The floor
        is a guard against the model under-delivering, not a minimum product."""
        from app.agent.validators import validate_plan

        plan, profile, targets = self._plan_and_profile(days)
        assert validate_plan(plan, profile, targets, expected_days=days).is_valid


class TestConfigurablePlanLength:
    """Plan length is a setting, not a constant.

    Seven days costs roughly 7500 tokens of an 8000-per-minute free tier,
    leaving nothing for the retry the agent is designed around — so the default
    is four, and anyone whose provider allows more can raise it without a code
    change.
    """

    def test_the_graph_follows_the_setting(self):
        from app.agent.graph import PLAN_DURATION_DAYS
        from app.core.config import settings

        assert PLAN_DURATION_DAYS == settings.plan_duration_days

    # Groq's free tier, the tightest limit this runs on. Prompt measured from
    # the real builders and rounded up to cover the tool schema they carry.
    FREE_TIER_TPM = 8000
    PROMPT = 1400

    def test_no_single_request_can_be_refused_outright(self):
        """The invariant that must never break, at any profile size.

        A 429 is waited out by `_RateLimitRetrying`; a 413 is not — a provider
        refuses a request whose reservation alone crowds the per-minute limit,
        and that refusal is final. So the retry escalation has a hard ceiling:
        growth that cures a truncation by asking for 7000 tokens would trade a
        recoverable failure for a fatal one.
        """
        from app.agent.llm import budget_for
        from app.models.plan import MealPlanDraft

        # Every profile the onboarding form allows, at the last retry.
        for meals_per_day in range(2, 7):
            worst = self.PROMPT + budget_for(
                MealPlanDraft, meals_per_day=meals_per_day, attempt=3
            )
            assert worst < self.FREE_TIER_TPM, (
                f"{meals_per_day} meals/day reserves {worst} tokens on its last "
                f"attempt against a {self.FREE_TIER_TPM} limit — the provider "
                "will refuse it with a 413, which is not retried"
            )

    def test_the_first_attempt_fits_in_one_minute_at_the_default_profile(self):
        """Both specialists share a superstep, so they share a rate window.

        This used to demand that a first attempt *and* a retry fit the same
        minute. That goal is gone, and deliberately: meeting it meant reserving
        1900 tokens for a plan whose JSON measures 1631, leaving nothing for the
        model to think with — so the first attempt truncated every time and the
        retry it had saved room for was spent re-failing. Fitting two cheap
        attempts is worth nothing next to one that works.

        A larger profile can still spill past the limit and wait on the rate
        limiter. That is a handled, recoverable second or two; truncation was
        neither.
        """
        from app.agent import graph
        from app.agent.llm import budget_for
        from app.models.plan import MealPlanDraft, TrainingPlanDraft

        chunks = graph.meal_chunk_count()
        first = chunks * (self.PROMPT + budget_for(MealPlanDraft)) + (
            self.PROMPT + budget_for(TrainingPlanDraft)
        )

        assert first <= self.FREE_TIER_TPM, (
            f"the default profile's first attempt reserves {first} tokens "
            f"against a {self.FREE_TIER_TPM} limit — the happy path would be "
            "rate limited before it ever succeeded"
        )

    def test_a_retry_fits_on_its_own(self):
        """The trainer reuses its draft on a retry, so only the meals re-run."""
        from app.agent import graph
        from app.agent.llm import budget_for
        from app.models.plan import MealPlanDraft

        chunks = graph.meal_chunk_count()
        retry = chunks * (self.PROMPT + budget_for(MealPlanDraft, attempt=2))

        assert retry <= self.FREE_TIER_TPM

    def test_the_stored_plan_records_its_own_length(self):
        """A plan generated under one setting must keep working if the setting
        changes — `_resolve_plan_day` counts against the plan, not the config."""
        from tests.factories import make_health_plan, make_targets

        plan = make_health_plan(make_targets(), days=4)
        assert len(plan.daily_plans) == 4

    @pytest.mark.parametrize("days", [1, 4, 7, 14])
    def test_any_supported_length_validates(self, days):
        from app.agent.validators import validate_plan
        from app.models.enums import DietType
        from tests.factories import make_health_plan, make_profile, make_targets

        targets = make_targets()
        plan = make_health_plan(targets, days=days)
        profile = make_profile(diet_type=DietType.VEGETARIAN)

        assert validate_plan(plan, profile, targets, expected_days=days).is_valid
