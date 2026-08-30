"""Meal and metric logging. The sensing surface of the agent loop."""

from datetime import date, timedelta
from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.api.deps import CurrentProfile, CurrentUser
from app.core.logging import get_logger
from app.db.repositories import LogRepository, PlanRepository
from app.models.enums import MealStatus
from app.models.log import (
    AdherenceSnapshot,
    DailyLogInDB,
    DailyMetricsRequest,
    MealLogEntry,
    MealLogRequest,
    SessionLogEntry,
    SessionLogRequest,
)
from app.services.adherence import build_snapshot
from app.services.nutrition import calculate_targets

router = APIRouter(prefix="/logs", tags=["logs"])
logger = get_logger(__name__)

# Consecutive skipped sessions before the training plan is treated as the
# problem. Three, matching the meal rule: two is a bad week, three is a habit.
STRUCTURAL_SESSION_SKIP_STREAK = 3


class MealLogResponse(BaseModel):
    """Confirms the log and tells the frontend whether to run the agent.

    Returning `agent_recommended` here saves the client from re-deriving the
    trigger conditions. The rules live in one place, on the server.
    """

    log: DailyLogInDB
    snapshot: AdherenceSnapshot
    agent_recommended: bool
    agent_reason: Optional[str] = None


class WeightPoint(BaseModel):
    date: date
    weight_kg: float


@router.post("/meals", response_model=MealLogResponse)
async def log_meal(
    payload: MealLogRequest,
    user: CurrentUser,
    profile: CurrentProfile,
    log_date: Optional[date] = Query(default=None),
) -> MealLogResponse:
    """Record what happened with a planned meal.

    A skip logged here is the primary trigger for adaptive replanning.
    """
    user_id = str(user.id)
    target_date = log_date or date.today()

    entry = MealLogEntry(
        meal_id=payload.meal_id,
        status=payload.status,
        actual_calories_kcal=payload.actual_calories_kcal,
        actual_protein_g=payload.actual_protein_g,
        substitute_name=payload.substitute_name,
        note=payload.note,
    )
    log = await LogRepository.upsert_meal(user_id, target_date, entry)

    targets = calculate_targets(profile)
    snapshot = build_snapshot(
        target_date=target_date,
        targets=targets,
        plan=await PlanRepository.get_active(user_id),
        today_log=log,
        recent_logs=await LogRepository.get_recent(user_id, days=7),
    )

    recommended, reason = _should_suggest_agent(payload.status, snapshot)

    return MealLogResponse(
        log=log,
        snapshot=snapshot,
        agent_recommended=recommended,
        agent_reason=reason,
    )


@router.post("/sessions", response_model=MealLogResponse)
async def log_session(
    payload: SessionLogRequest,
    user: CurrentUser,
    profile: CurrentProfile,
    log_date: Optional[date] = Query(default=None),
) -> MealLogResponse:
    """Record whether the day's training happened.

    The counterpart to logging a meal, and it exists for the same reason: the
    plan can only adapt to what it is told. Skipping sessions used to be
    invisible, so a training plan nobody could follow was rewritten only when
    the food happened to be wrong too.
    """
    user_id = str(user.id)
    target_date = log_date or date.today()

    log = await LogRepository.upsert_session(
        user_id,
        target_date,
        SessionLogEntry(
            plan_day=payload.plan_day, status=payload.status, note=payload.note
        ),
    )

    snapshot = build_snapshot(
        target_date=target_date,
        targets=calculate_targets(profile),
        plan=await PlanRepository.get_active(user_id),
        today_log=log,
        recent_logs=await LogRepository.get_recent(user_id, days=7),
    )

    recommended, reason = _should_suggest_agent_after_session(snapshot)

    return MealLogResponse(
        log=log,
        snapshot=snapshot,
        agent_recommended=recommended,
        agent_reason=reason,
    )


def _should_suggest_agent_after_session(
    snapshot: AdherenceSnapshot,
) -> tuple[bool, Optional[str]]:
    """One skipped session is a Tuesday. Three is a plan that does not fit.

    Deliberately quieter than the meal rule. Missing a workout has no knock-on
    effect on the rest of the day, so there is nothing to rebalance and
    offering to replan after a single skip would be nagging.
    """
    if snapshot.session_skip_streak_days >= STRUCTURAL_SESSION_SKIP_STREAK:
        return True, (
            f"You've skipped training {snapshot.session_skip_streak_days} days "
            "running. Kaya can rebuild the week around sessions you'll "
            "actually do."
        )
    return False, None


def _should_suggest_agent(
    status: MealStatus, snapshot: AdherenceSnapshot
) -> tuple[bool, Optional[str]]:
    """Would running the agent right now actually change anything?"""
    if status == MealStatus.SKIPPED and snapshot.meals_pending > 0:
        return True, (
            f"You have {snapshot.calories_remaining} kcal and "
            f"{snapshot.protein_remaining_g}g protein left across "
            f"{snapshot.meals_pending} remaining meal(s). Kaya can rebalance them."
        )

    if snapshot.skip_streak_days >= 3:
        return True, (
            f"You've skipped meals {snapshot.skip_streak_days} days running. "
            "Kaya can restructure the plan to fit your routine better."
        )

    if snapshot.calories_consumed > snapshot.calories_target * 1.15 and (
        snapshot.meals_pending > 0
    ):
        return True, "You're over today's target with meals still to come."

    return False, None


@router.post("/metrics", response_model=DailyLogInDB)
async def log_metrics(
    payload: DailyMetricsRequest,
    user: CurrentUser,
    log_date: Optional[date] = Query(default=None),
) -> DailyLogInDB:
    """Record weight, steps, sleep and water for a day."""
    return await LogRepository.update_metrics(
        str(user.id),
        log_date or date.today(),
        payload.model_dump(exclude_none=True),
    )


@router.get("/today", response_model=DailyLogInDB)
async def get_today(
    user: CurrentUser, log_date: Optional[date] = Query(default=None)
) -> DailyLogInDB:
    return await LogRepository.get_or_create(str(user.id), log_date or date.today())


@router.get("/adherence", response_model=AdherenceSnapshot)
async def get_adherence(
    user: CurrentUser,
    profile: CurrentProfile,
    log_date: Optional[date] = Query(default=None),
) -> AdherenceSnapshot:
    """The current adherence picture. Computed, never model-generated."""
    user_id = str(user.id)
    target_date = log_date or date.today()

    return build_snapshot(
        target_date=target_date,
        targets=calculate_targets(profile),
        plan=await PlanRepository.get_active(user_id),
        today_log=await LogRepository.get_or_create(user_id, target_date),
        recent_logs=await LogRepository.get_recent(user_id, days=7),
    )


@router.get("/history", response_model=List[DailyLogInDB])
async def get_history(user: CurrentUser, days: int = Query(default=30, ge=1, le=365)):
    return await LogRepository.get_recent(str(user.id), days=days)


@router.get("/weight", response_model=List[WeightPoint])
async def get_weight_series(
    user: CurrentUser, days: int = Query(default=90, ge=7, le=365)
) -> List[WeightPoint]:
    """Real logged weights for the progress chart.

    Note this returns only what the user actually recorded. The Streamlit
    version generated a synthetic trend line, which looked good and meant
    nothing. An empty list here is honest.
    """
    logs = await LogRepository.get_range(
        str(user.id), date.today() - timedelta(days=days - 1), date.today()
    )
    return [
        WeightPoint(date=log.log_date, weight_kg=log.weight_kg)
        for log in logs
        if log.weight_kg is not None
    ]
