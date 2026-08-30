"""What the person waiting for dinner is shown.

Two audiences read the same failure. Whoever can fix it needs the setting
name, the token budget and the HTTP status. The person who asked for a meal
plan can act on none of that, and showing them the machinery makes a working
product feel broken. "the model did not finish the structured output" is a
sentence no user should ever meet.

So the technical message goes to the logs and to development, and production
gets plain English.
"""

import pytest

from app.agent.llm import LLMFailure, failure_text
from app.core.config import settings


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class TestTheTechnicalMessageIsStillWritten:
    """Hiding it from users must not mean losing it."""

    def test_development_still_shows_the_diagnosis(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "development")
        failure = LLMFailure("LLM_MODEL is set to 'gemini-3.6-flash'", retryable=False)

        assert failure_text(failure) == failure.message

    def test_the_message_survives_on_the_object(self):
        """`public` is an addition, not a replacement. The log still needs it."""
        failure = LLMFailure("reserved 1900 tokens, needed 2710", retryable=True)

        assert "1900" in failure.message
        assert "1900" not in failure.public


class TestProductionSpeaksPlainly:
    @pytest.fixture(autouse=True)
    def _production(self, monkeypatch):
        monkeypatch.setattr(settings, "environment", "production")

    @pytest.mark.parametrize(
        "technical",
        [
            "The model did not finish the structured output the plan needs: it "
            "stopped part-way and what arrived could not be read.",
            "LLM_MAX_TOKENS in backend/.env caps how much more it can reserve.",
            "Rate limited by the provider. Kaya waits and retries on its own.",
        ],
    )
    def test_no_machinery_reaches_the_timeline(self, technical):
        shown = failure_text(LLMFailure(technical, retryable=True))

        for jargon in (
            "token",
            "LLM_",
            "structured output",
            "backend/.env",
            "max_tokens",
            "API",
            "HTTP",
        ):
            assert jargon.lower() not in shown.lower(), (
                f"'{jargon}' leaked into a message shown to a user: {shown!r}"
            )

    def test_a_retryable_failure_says_it_is_trying_again(self):
        """Because it is. A user watching the timeline should see effort, not
        an error that appears final."""
        shown = failure_text(LLMFailure("tool_use_failed", retryable=True))

        assert "again" in shown.lower()

    def test_a_final_failure_says_the_existing_plan_is_safe(self):
        """The first question anyone has when generation fails is whether they
        just lost what they had."""
        shown = failure_text(LLMFailure("model_not_found", retryable=False))

        assert "unchanged" in shown.lower()

    def test_it_never_blames_the_user(self):
        shown = failure_text(LLMFailure("invalid api key", retryable=False))

        for accusation in ("you did", "your mistake", "invalid", "error:"):
            assert accusation not in shown.lower()


class TestTheTimelineIsNotALog:
    """What the run trace says when a plan is rejected and redrawn.

    A rejection is the product working: the deterministic validator caught
    something the model got wrong and sent it back. Shown as
    "Rejected: Day 2 totals 2010 kcal, outside the acceptable range 2012-2560"
    it reads as a fault instead, and names an acceptance range nobody outside
    this repository has any use for.
    """

    from app.agent.validators import ValidationResult, summarise_for_user

    def _summary(self, *errors):
        from app.agent.validators import ValidationResult, summarise_for_user

        return summarise_for_user(ValidationResult(errors=list(errors)))

    def test_a_diet_violation_is_named_without_the_keyword_machinery(self):
        summary = self._summary(
            "Day 1 breakfast contains 'egg', which is forbidden for a Jain diet."
        )

        assert "don't eat" in summary
        assert "forbidden" not in summary
        assert "'egg'" not in summary

    def test_a_calorie_miss_does_not_quote_the_acceptance_range(self):
        summary = self._summary(
            "Day 2 totals 2010 kcal, short of the acceptable range 2012-2560 "
            "for a 2286 kcal target. Day 2 needs 2 kcal MORE across the day."
        )

        assert "calories" in summary
        for machinery in ("2010", "2012", "range", "kcal"):
            assert machinery not in summary

    def test_it_says_what_happened_next(self):
        """Otherwise it reads as a dead end rather than a step in the process."""
        summary = self._summary("Day 3 provides only 90g protein, below the 120g.")

        assert "redrawn" in summary.lower() or "again" in summary.lower()

    def test_several_problems_are_one_sentence(self):
        summary = self._summary(
            "Day 1 breakfast contains 'egg', which is forbidden for a Jain diet.",
            "Day 2 totals 2010 kcal, short of the acceptable range 2012-2560.",
        )

        assert summary.count(".") <= 2, f"reads like a list, not a sentence: {summary}"


class TestTheDeltaIsAlwaysActionable:
    """"Needs about 0 kcal MORE in every meal" is not an instruction."""

    def test_a_tiny_miss_asks_for_the_day_not_the_meal(self):
        from app.agent.validators import _check_day_totals, ValidationResult
        from tests.factories import make_meal_draft, make_targets

        targets = make_targets(calories=2286, protein=140)
        draft = make_meal_draft(targets, days=1, meals_per_day=4)
        day = draft.days[0]

        # Shave the day to two calories under the floor.
        floor = targets.calories_kcal * 0.88
        total = sum(m.calories_kcal for m in day.meals)
        day.meals[0].calories_kcal -= int(total - floor) + 2

        result = ValidationResult()

        class _Day:
            def __init__(self, inner):
                self.day = 1
                self.meals = inner.meals

        _check_day_totals(_Day(day), targets, result)

        assert result.errors, "a day under the floor should still be rejected"
        assert "about 0 kcal" not in result.errors[0], result.errors[0]
        assert "across the day" in result.errors[0], result.errors[0]
