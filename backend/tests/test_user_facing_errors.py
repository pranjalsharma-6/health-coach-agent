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
