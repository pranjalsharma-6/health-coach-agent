"""End-to-end API tests against an in-memory Mongo.

Covers the whole request path. Auth, dependency injection, repositories,
serialisation, without needing a database server or an LLM. The agent's LLM
call is stubbed; everything around it is real.
"""

import re

import pytest
from httpx import ASGITransport, AsyncClient
from mongomock_motor import AsyncMongoMockClient

from app.db import mongo
from app.main import app
from app.models.plan import HealthPlan
from tests.factories import (
    make_health_plan,
    make_targets,
    scope_to_requested_days,
)

BASE = "/api/v1"


@pytest.fixture(autouse=True)
async def in_memory_db(monkeypatch):
    """Point the app's database handle at a mock and reset it per test."""
    client = AsyncMongoMockClient()
    database = client["KayaTestDB"]

    monkeypatch.setattr(mongo, "_client", client)
    monkeypatch.setattr(mongo, "_database", database)
    monkeypatch.setattr(mongo, "get_database", lambda: database)
    # Repositories imported `get_database` by name, so patch it there too.
    monkeypatch.setattr("app.db.repositories.get_database", lambda: database)

    yield database


@pytest.fixture
async def client():
    """An HTTP client bound to the ASGI app, bypassing the network."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def register(client, email="test@example.com") -> str:
    """Create an account and return its bearer token."""
    response = await client.post(
        f"{BASE}/auth/register",
        json={"email": email, "password": "supersecret123", "full_name": "Test User"},
    )
    assert response.status_code == 201, response.text
    return response.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


PROFILE_PAYLOAD = {
    "gender": "female",
    "age_years": 28,
    "height_cm": 165,
    "current_weight_kg": 72.0,
    "target_weight_kg": 64.0,
    "goal": "fat_loss",
    "activity_level": "lightly_active",
    "target_timeline_weeks": 16,
    "diet_type": "eggetarian",
    "cuisine_preferences": ["south_indian"],
    "allergies": ["Peanut", "peanut", " "],
    "disliked_foods": ["bitter gourd"],
    "meals_per_day": 4,
    "cooking_skill": "beginner",
    "max_prep_minutes": 25,
    "budget_tier": "low",
    "eat_out_per_week": 3,
    "medical_notes": None,
}


class TestAuth:
    async def test_register_returns_a_usable_token(self, client):
        token = await register(client)

        response = await client.get(f"{BASE}/auth/me", headers=auth(token))
        assert response.status_code == 200
        assert response.json()["email"] == "test@example.com"
        assert response.json()["onboarded"] is False

    async def test_duplicate_email_is_rejected(self, client):
        await register(client)
        response = await client.post(
            f"{BASE}/auth/register",
            json={
                "email": "test@example.com",
                "password": "supersecret123",
                "full_name": "Someone Else",
            },
        )
        assert response.status_code == 409

    async def test_login_with_correct_password_succeeds(self, client):
        await register(client)
        response = await client.post(
            f"{BASE}/auth/login",
            json={"email": "test@example.com", "password": "supersecret123"},
        )
        assert response.status_code == 200
        assert response.json()["access_token"]

    async def test_wrong_password_is_rejected(self, client):
        await register(client)
        response = await client.post(
            f"{BASE}/auth/login",
            json={"email": "test@example.com", "password": "wrongpassword"},
        )
        assert response.status_code == 401

    async def test_unknown_email_gives_the_same_error_as_a_wrong_password(
        self, client
    ):
        """Don't leak which emails have accounts."""
        await register(client)
        wrong_password = await client.post(
            f"{BASE}/auth/login",
            json={"email": "test@example.com", "password": "wrongpassword"},
        )
        unknown_user = await client.post(
            f"{BASE}/auth/login",
            json={"email": "nobody@example.com", "password": "supersecret123"},
        )
        assert wrong_password.status_code == unknown_user.status_code == 401
        assert wrong_password.json()["detail"] == unknown_user.json()["detail"]

    async def test_protected_routes_reject_missing_tokens(self, client):
        response = await client.get(f"{BASE}/profile")
        assert response.status_code == 401

    async def test_protected_routes_reject_garbage_tokens(self, client):
        response = await client.get(f"{BASE}/profile", headers=auth("not-a-jwt"))
        assert response.status_code == 401

    async def test_password_hash_is_never_returned(self, client):
        token = await register(client)
        body = await client.get(f"{BASE}/auth/me", headers=auth(token))
        assert "hashed_password" not in body.text


class TestProfileAndTargets:
    async def test_onboarding_creates_a_profile_and_flips_the_flag(self, client):
        token = await register(client)

        response = await client.post(
            f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token)
        )
        assert response.status_code == 201, response.text

        me = await client.get(f"{BASE}/auth/me", headers=auth(token))
        assert me.json()["onboarded"] is True

    async def test_allergies_are_normalised(self, client):
        """Duplicates, casing and blanks are cleaned before storage."""
        token = await register(client)
        response = await client.post(
            f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token)
        )
        assert response.json()["allergies"] == ["peanut"]

    async def test_targets_are_computed_from_the_profile(self, client):
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        response = await client.get(f"{BASE}/profile/targets", headers=auth(token))
        assert response.status_code == 200

        body = response.json()
        assert body["bmr_kcal"] > 0
        assert body["targets"]["calories_kcal"] >= 1200
        # Fat loss goal: protein at 2.0 g/kg of 72kg.
        assert body["targets"]["protein_g"] == 144
        assert body["estimated_weekly_change_kg"] < 0
        assert sum(body["macro_split_percent"].values()) == pytest.approx(100, abs=2)

    async def test_endpoints_needing_a_profile_return_409_before_onboarding(
        self, client
    ):
        token = await register(client)
        response = await client.get(f"{BASE}/profile/targets", headers=auth(token))
        assert response.status_code == 409

    async def test_partial_update_only_changes_supplied_fields(self, client):
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        response = await client.patch(
            f"{BASE}/profile",
            json={"diet_type": "vegan", "current_weight_kg": 70.5},
            headers=auth(token),
        )
        assert response.status_code == 200

        body = response.json()
        assert body["diet_type"] == "vegan"
        assert body["current_weight_kg"] == 70.5
        assert body["cuisine_preferences"] == ["south_indian"]  # untouched

    async def test_invalid_profile_values_are_rejected(self, client):
        token = await register(client)
        response = await client.post(
            f"{BASE}/profile",
            json={**PROFILE_PAYLOAD, "age_years": 4},
            headers=auth(token),
        )
        assert response.status_code == 422


class TestAgentRun:
    async def test_agent_run_creates_a_plan(self, client, monkeypatch):
        """The full graph, with only the LLM call stubbed."""
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )

        _stub_llm(monkeypatch, make_health_plan(targets))

        response = await client.post(f"{BASE}/agent/run", headers=auth(token))
        assert response.status_code == 200, response.text

        body = response.json()
        assert body["decision"] == "create_initial"
        assert body["error"] is None
        assert body["plan"] is not None
        # Length is configurable; assert against the configuration rather
        # than a number that used to be true.
        from app.agent.graph import PLAN_DURATION_DAYS

        assert len(body["plan"]["daily_plans"]) == PLAN_DURATION_DAYS
        assert body["plan"]["version"] == 1

    async def test_active_plan_is_retrievable_after_a_run(self, client, monkeypatch):
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )
        _stub_llm(monkeypatch, make_health_plan(targets))

        await client.post(f"{BASE}/agent/run", headers=auth(token))

        response = await client.get(f"{BASE}/plans/active", headers=auth(token))
        assert response.status_code == 200
        assert response.json()["is_active"] is True

    async def test_agent_records_an_event_even_for_a_no_op(
        self, client, monkeypatch
    ):
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )
        _stub_llm(monkeypatch, make_health_plan(targets))

        await client.post(f"{BASE}/agent/run", headers=auth(token))
        # Second run: a plan exists and nothing has been logged, so no action.
        second = await client.post(f"{BASE}/agent/run", headers=auth(token))
        assert second.json()["decision"] == "no_action"

        events = await client.get(f"{BASE}/agent/events", headers=auth(token))
        decisions = [e["decision"] for e in events.json()]
        assert "create_initial" in decisions
        assert "no_action" in decisions

    async def test_a_plan_violating_the_diet_is_rejected_and_retried(
        self, client, monkeypatch
    ):
        """The validator must catch an off-diet plan and force a regeneration."""
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )

        # First attempt serves chicken to an eggetarian; second is clean.
        bad_plan = make_health_plan(targets)
        bad_plan.daily_plans[0].meals[0].name = "Chicken sandwich"
        good_plan = make_health_plan(targets)

        _stub_llm(monkeypatch, bad_plan, good_plan)

        response = await client.post(f"{BASE}/agent/run", headers=auth(token))
        body = response.json()

        assert body["attempts"] == 2, "should have regenerated after rejection"
        assert body["plan"] is not None

        # The *saved* plan must be clean. The step trace legitimately quotes the
        # rejection reason, so assert on the plan rather than the whole response.
        meal_names = [
            meal["name"].lower()
            for day in body["plan"]["daily_plans"]
            for meal in day["meals"]
        ]
        assert not any("chicken" in name for name in meal_names)

        # And the rejection is visible in the trace, so the retry is auditable.
        assert any(
            step["status"] == "failed" and "chicken" in str(step).lower()
            for step in body["steps"]
        )

    async def test_run_fails_cleanly_when_every_attempt_is_invalid(
        self, client, monkeypatch
    ):
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )

        bad = make_health_plan(targets)
        bad.daily_plans[0].meals[0].name = "Chicken sandwich"
        _stub_llm(monkeypatch, bad, bad, bad)

        response = await client.post(f"{BASE}/agent/run", headers=auth(token))
        body = response.json()

        assert body["plan"] is None
        assert body["error"] is not None
        assert "unchanged" in body["error"]

        # No junk written to the database.
        plans = await client.get(f"{BASE}/plans/history", headers=auth(token))
        assert plans.json() == []


class TestLogging:
    async def test_logging_a_skip_recommends_running_the_agent(
        self, client, monkeypatch
    ):
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )
        _stub_llm(monkeypatch, make_health_plan(targets))
        await client.post(f"{BASE}/agent/run", headers=auth(token))

        response = await client.post(
            f"{BASE}/logs/meals",
            json={"meal_id": "d1-breakfast", "status": "skipped"},
            headers=auth(token),
        )
        assert response.status_code == 200, response.text

        body = response.json()
        assert body["snapshot"]["meals_skipped"] == 1
        assert body["agent_recommended"] is True
        assert body["agent_reason"]

    async def test_relogging_a_meal_corrects_rather_than_duplicates(
        self, client, monkeypatch
    ):
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )
        _stub_llm(monkeypatch, make_health_plan(targets))
        await client.post(f"{BASE}/agent/run", headers=auth(token))

        for status in ("skipped", "eaten"):
            await client.post(
                f"{BASE}/logs/meals",
                json={"meal_id": "d1-breakfast", "status": status},
                headers=auth(token),
            )

        log = await client.get(f"{BASE}/logs/today", headers=auth(token))
        entries = [m for m in log.json()["meals"] if m["meal_id"] == "d1-breakfast"]

        assert len(entries) == 1
        assert entries[0]["status"] == "eaten"

    async def test_metrics_are_stored_and_surface_in_the_weight_series(
        self, client
    ):
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        await client.post(
            f"{BASE}/logs/metrics",
            json={"weight_kg": 71.4, "steps": 9200, "sleep_hours": 7.5},
            headers=auth(token),
        )

        series = await client.get(f"{BASE}/logs/weight", headers=auth(token))
        assert series.status_code == 200
        assert series.json() == [
            {"date": series.json()[0]["date"], "weight_kg": 71.4}
        ]

    async def test_weight_series_is_empty_rather_than_synthetic(self, client):
        """No fabricated trend line. An empty chart is the honest answer."""
        token = await register(client)
        await client.post(f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(token))

        series = await client.get(f"{BASE}/logs/weight", headers=auth(token))
        assert series.json() == []


class TestTenantIsolation:
    async def test_one_user_cannot_read_another_users_plan(
        self, client, monkeypatch
    ):
        alice_token = await register(client, "alice@example.com")
        await client.post(
            f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(alice_token)
        )

        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(alice_token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )
        _stub_llm(monkeypatch, make_health_plan(targets))
        run = await client.post(f"{BASE}/agent/run", headers=auth(alice_token))
        alice_plan_id = run.json()["plan"]["_id"]

        bob_token = await register(client, "bob@example.com")
        response = await client.get(
            f"{BASE}/plans/{alice_plan_id}", headers=auth(bob_token)
        )
        assert response.status_code == 404

    async def test_each_user_sees_only_their_own_active_plan(
        self, client, monkeypatch
    ):
        alice_token = await register(client, "alice@example.com")
        await client.post(
            f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(alice_token)
        )
        targets_response = await client.get(
            f"{BASE}/profile/targets", headers=auth(alice_token)
        )
        targets = make_targets(
            calories=targets_response.json()["targets"]["calories_kcal"],
            protein=targets_response.json()["targets"]["protein_g"],
        )
        _stub_llm(monkeypatch, make_health_plan(targets))
        await client.post(f"{BASE}/agent/run", headers=auth(alice_token))

        bob_token = await register(client, "bob@example.com")
        await client.post(
            f"{BASE}/profile", json=PROFILE_PAYLOAD, headers=auth(bob_token)
        )

        response = await client.get(f"{BASE}/plans/active", headers=auth(bob_token))
        assert response.json() is None


class TestMeta:
    async def test_health_endpoint_reports_status(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert "database" in response.json()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _stub_llm(monkeypatch, *plans: HealthPlan) -> None:
    """Replace each specialist's LLM with one returning fixed output.

    Takes whole `HealthPlan`s for readability at the call site, then splits each
    into the nutritionist's and trainer's halves. The graph asks for those two
    schemas separately now. The last plan repeats if the agent takes more
    attempts than provided.
    """
    from app.models.plan import (
        DayMeals,
        ActivityDraft,
        DayTraining,
        ExerciseDraft,
        MealDraftItem,
        MealPlanDraft,
        PlanCritique,
        TrainingPlanDraft,
    )

    meal_queue = [
        MealPlanDraft(
            plan_title=plan.plan_title,
            reasoning=plan.agent_reasoning,
            days=[
                DayMeals(
                    day=d.day,
                    theme=d.theme,
                    meals=[MealDraftItem.from_meal_item(m) for m in d.meals],
                )
                for d in plan.daily_plans
            ],
        )
        for plan in plans
    ]
    training = TrainingPlanDraft(
        reasoning="Alternating strength and easy movement with a mid-week rest day.",
        days=[
            DayTraining(
                day=d.day,
                activity=ActivityDraft(
                    activity_type=d.activity.activity_type,
                    duration_minutes=d.activity.duration_minutes,
                    intensity=d.activity.intensity,
                    description=d.activity.description,
                    # Carried through: a non-rest day with no exercises is a
                    # validation error, so dropping them here would reject
                    # every plan these tests build.
                    exercises=[
                        ExerciseDraft(
                            name=e.name,
                            sets=e.sets,
                            reps=e.reps,
                            rest_seconds=e.rest_seconds,
                        )
                        for e in d.activity.exercises
                    ],
                ),
            )
            for d in plans[0].daily_plans
        ],
    )
    def _requested_days(messages):
        text = "\n".join(str(getattr(m, "content", m)) for m in messages)
        match = re.search(r"DAYS (\d+) TO (\d+) ONLY", text)
        return (int(match.group(1)), int(match.group(2))) if match else None

    approving_critic = PlanCritique(
        approved=True, issues=[], summary="The week is coherent."
    )

    # The nutritionist drafts the week in chunks, so one generation round is
    # several calls. The queue has to advance once per round, not once per
    # call, or a "reject the first plan, accept the second" test silently
    # consumes both plans on its first attempt.
    current: dict = {"draft": None}

    class StubStructuredLLM:
        def __init__(self, schema):
            self._schema = schema

        async def ainvoke(self, messages):
            if self._schema is TrainingPlanDraft:
                return training
            if self._schema is PlanCritique:
                return approving_critic

            window = _requested_days(messages)
            if window is None or window[0] == 1:
                current["draft"] = (
                    meal_queue.pop(0) if len(meal_queue) > 1 else meal_queue[0]
                )
            return scope_to_requested_days(current["draft"], messages)

    monkeypatch.setattr(
        "app.agent.graph.get_structured_llm",
        lambda schema, **_budget: StubStructuredLLM(schema),
    )
    monkeypatch.setattr("app.agent.llm.is_configured", lambda: True)
    monkeypatch.setattr("app.api.routes.agent.is_configured", lambda: True)
