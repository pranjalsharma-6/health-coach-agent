"""Tests for settings loading.

`cors_origins` has a history: it was typed as a plain `List[str]`, which made
pydantic-settings JSON-decode the raw environment value inside the env source,
before any validator ran. The comma-separated form printed in `.env.example`
and in the deployment guide therefore crashed the app at import time with an
opaque `SettingsError`. These tests pin every shape the value actually arrives
in — from a `.env` file locally, and from a real environment variable on Render.
"""

import pathlib

import pytest

from app.core.config import Settings

LOCAL_DEFAULT = ["http://localhost:3000", "http://127.0.0.1:3000"]


@pytest.fixture
def from_env(monkeypatch):
    """Load settings from a real environment variable, as a host would set it."""

    def load(raw: str | None) -> Settings:
        if raw is None:
            monkeypatch.delenv("CORS_ORIGINS", raising=False)
        else:
            monkeypatch.setenv("CORS_ORIGINS", raw)
        return Settings(_env_file=None)

    return load


@pytest.fixture
def from_dotenv(tmp_path, monkeypatch):
    """Load settings from a .env file, as a developer would write it."""

    def load(raw: str) -> Settings:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        env = tmp_path / ".env"
        env.write_text(f"CORS_ORIGINS={raw}\n", encoding="utf-8")
        return Settings(_env_file=env)

    return load


COMMA_SEPARATED = [
    # The line in .env.example, verbatim.
    ("http://localhost:3000,http://127.0.0.1:3000", LOCAL_DEFAULT),
    # One origin, which is what the deployment guide has you paste into Render.
    ("https://kaya.vercel.app", ["https://kaya.vercel.app"]),
    ("https://a.com, https://b.com", ["https://a.com", "https://b.com"]),
    ("https://a.com,", ["https://a.com"]),
    ("  https://a.com  ", ["https://a.com"]),
    ("", []),
]


class TestCorsOrigins:
    @pytest.mark.parametrize("raw,expected", COMMA_SEPARATED)
    def test_comma_separated_from_a_dotenv_file(self, from_dotenv, raw, expected):
        assert from_dotenv(raw).cors_origins == expected

    @pytest.mark.parametrize("raw,expected", COMMA_SEPARATED)
    def test_comma_separated_from_an_environment_variable(
        self, from_env, raw, expected
    ):
        """Render sets a real env var, not a .env file — a different source
        class, and it decodes complex fields too."""
        assert from_env(raw).cors_origins == expected

    def test_json_array_still_works(self, from_env):
        """The shape the old list-typed field required. Someone's existing
        deployment may still be set this way; it must not break."""
        assert from_env('["https://a.com", "https://b.com"]').cors_origins == [
            "https://a.com",
            "https://b.com",
        ]

    def test_malformed_json_does_not_crash_the_app(self, from_env):
        """A bad value should cost you CORS, not the whole service."""
        assert from_env("[https://a.com").cors_origins == ["[https://a.com"]

    def test_unset_falls_back_to_local_development(self, from_env):
        assert from_env(None).cors_origins == LOCAL_DEFAULT

    def test_a_list_passes_through_untouched(self):
        assert Settings(_env_file=None, cors_origins=["https://a.com"]).cors_origins == [
            "https://a.com"
        ]


def test_the_shipped_env_example_actually_loads(tmp_path, monkeypatch):
    """The strongest version of this test: load `.env.example` itself.

    If someone adds another list-typed setting and writes a comma-separated
    example for it, this fails before a beginner hits it on their first run.
    """
    example = pathlib.Path(__file__).parent.parent / ".env.example"
    assert example.exists(), "backend/.env.example is missing"

    for key in ("CORS_ORIGINS", "MONGODB_URI", "GROQ_API_KEY", "JWT_SECRET"):
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=example)

    assert settings.cors_origins == LOCAL_DEFAULT
    assert settings.mongodb_db_name == "KayaDB"


class TestCuisinePreferences:
    """Cuisine went from one enum to a list.

    People eat across cuisines — North Indian on weekdays, continental at the
    weekend — and forcing one choice made the plan less like their actual food,
    which is the whole adherence lever. Profiles written before the change hold
    a bare string under the old key, and must keep working.
    """

    @staticmethod
    def _profile(**extra):
        from app.models.profile import ProfileInDB

        return ProfileInDB.model_validate(
            {
                "user_id": "u",
                "full_name": "P",
                "age_years": 22,
                "gender": "male",
                "goal": "muscle_gain",
                "diet_type": "vegetarian",
                "height_cm": 175,
                "current_weight_kg": 70,
                **extra,
            }
        )

    def test_reads_a_profile_written_before_the_change(self):
        """The old key, holding a bare string. Without the alias this silently
        resets to "mixed" and the user loses a choice they made."""
        assert self._profile(cuisine_preference="south_indian").cuisine_preferences == [
            "south_indian"
        ]

    def test_accepts_several(self):
        assert self._profile(
            cuisine_preferences=["north_indian", "mediterranean"]
        ).cuisine_preferences == ["north_indian", "mediterranean"]

    def test_an_empty_selection_means_mixed(self):
        """Not an error — "no strong preference" is a real answer, and it is
        what MIXED encodes. Leaving it empty would strip cuisine guidance from
        the prompt entirely."""
        assert self._profile(cuisine_preferences=[]).cuisine_preferences == ["mixed"]

    def test_the_prompt_tells_the_model_not_to_fuse_them(self):
        """A bare list invites a miso rajma."""
        from app.agent.prompts import build_constraints_block

        block = build_constraints_block(
            self._profile(cuisine_preferences=["north_indian", "east_asian"])
        )
        assert "never blended into a single dish" in block
        assert "north indian" in block.lower()
        assert "east asian" in block.lower()

    def test_a_single_choice_reads_naturally(self):
        """One cuisine shouldn't get the multi-cuisine preamble."""
        from app.agent.prompts import build_constraints_block

        block = build_constraints_block(self._profile(cuisine_preferences=["south_indian"]))
        assert "several cuisines" not in block
        assert "idli" in block


class TestOptionalIntegerSettings:
    def test_a_blank_llm_max_tokens_means_unset(self, from_env):
        """Leaving an optional key empty is the obvious thing to do. Pydantic
        refuses "" as an int, so without coercion the app dies at import over
        a setting the user did not need."""
        assert from_env_int(from_env, "") is None

    def test_a_real_value_is_read(self, from_env):
        assert from_env_int(from_env, "1500") == 1500

    def test_unset_is_none(self, from_env):
        assert from_env_int(from_env, None) is None


def from_env_int(from_env, raw):
    import os

    if raw is None:
        os.environ.pop("LLM_MAX_TOKENS", None)
    else:
        os.environ["LLM_MAX_TOKENS"] = raw
    try:
        return Settings(_env_file=None).llm_max_tokens
    finally:
        os.environ.pop("LLM_MAX_TOKENS", None)
