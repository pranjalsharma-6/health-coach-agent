"""Adherence evaluation. The 'sense' and 'evaluate' halves of the agent loop.

This module answers "how is the user actually doing?" in plain Python, from
stored logs. The agent's decision is made from this evidence, so it must be
deterministic and reproducible: same logs in, same snapshot out, always.
"""

from datetime import date, timedelta
from typing import List, Optional

from app.core.logging import get_logger
from app.models.enums import MealStatus
from app.models.log import AdherenceSnapshot, DailyLogInDB
from app.models.plan import NutritionTargets, PlanInDB

logger = get_logger(__name__)

# How far over the calorie target counts as an overage worth reacting to.
CALORIE_OVERAGE_TOLERANCE = 1.15

# Below this 7-day adherence rate, the plan itself is the problem, not the user.
STRUCTURAL_ADHERENCE_THRESHOLD = 0.6


def build_snapshot(
    target_date: date,
    targets: NutritionTargets,
    plan: Optional[PlanInDB],
    today_log: Optional[DailyLogInDB],
    recent_logs: List[DailyLogInDB],
) -> AdherenceSnapshot:
    """Compute the day's adherence picture."""
    plan_day = _resolve_plan_day(plan, target_date)
    planned_meals = plan_day.meals if plan_day else []
    plan_day_number = plan_day.day if plan_day else None

    logged = {entry.meal_id: entry for entry in (today_log.meals if today_log else [])}

    eaten = skipped = 0
    calories_consumed = 0
    protein_consumed = 0

    for meal in planned_meals:
        entry = logged.get(meal.meal_id)
        if entry is None:
            continue

        if entry.status == MealStatus.SKIPPED:
            skipped += 1
        elif entry.status in (MealStatus.EATEN, MealStatus.SUBSTITUTED):
            eaten += 1
            # Prefer what they actually ate; fall back to what was planned.
            calories_consumed += (
                entry.actual_calories_kcal
                if entry.actual_calories_kcal is not None
                else meal.calories_kcal
            )
            protein_consumed += (
                entry.actual_protein_g
                if entry.actual_protein_g is not None
                else meal.protein_g
            )

    total_planned = len(planned_meals)
    pending = max(total_planned - eaten - skipped, 0)

    adherence_rate, meals_logged = _adherence_rate(recent_logs)

    return AdherenceSnapshot(
        date=target_date,
        plan_day=plan_day_number,
        meals_planned=total_planned,
        meals_eaten=eaten,
        meals_skipped=skipped,
        meals_pending=pending,
        calories_target=targets.calories_kcal,
        calories_consumed=calories_consumed,
        calories_remaining=targets.calories_kcal - calories_consumed,
        protein_target_g=targets.protein_g,
        protein_consumed_g=protein_consumed,
        protein_remaining_g=targets.protein_g - protein_consumed,
        steps=today_log.steps if today_log else None,
        sleep_hours=today_log.sleep_hours if today_log else None,
        skip_streak_days=_skip_streak(recent_logs, target_date),
        skips_last_7_days=_count_skips(recent_logs),
        adherence_rate_7d=adherence_rate,
        meals_logged_7d=meals_logged,
    )


def _resolve_plan_day(plan: Optional[PlanInDB], target_date: date):
    """Map a calendar date onto the right day of the plan.

    Plans are relative ('day 3'), not absolute, so we count days elapsed since
    the plan was created and wrap if the user has run past its duration.
    """
    if plan is None or not plan.daily_plans:
        return None

    days_elapsed = (target_date - plan.created_at.date()).days
    if days_elapsed < 0:
        return plan.daily_plans[0]

    index = days_elapsed % len(plan.daily_plans)
    return plan.daily_plans[index]


def _skip_streak(logs: List[DailyLogInDB], target_date: date) -> int:
    """Consecutive days ending today on which at least one meal was skipped."""
    by_date = {log.log_date: log for log in logs}
    streak = 0
    cursor = target_date

    while True:
        log = by_date.get(cursor)
        if log is None:
            break
        if not any(m.status == MealStatus.SKIPPED for m in log.meals):
            break
        streak += 1
        cursor -= timedelta(days=1)

    return streak


def _count_skips(logs: List[DailyLogInDB]) -> int:
    return sum(
        1 for log in logs for meal in log.meals if meal.status == MealStatus.SKIPPED
    )


def _adherence_rate(logs: List[DailyLogInDB]) -> tuple[float, int]:
    """Fraction of logged meals that were eaten, and the sample size behind it.

    Only counts meals the user actually logged. A day they never opened the app
    is missing data, not a failure. Treating silence as non-adherence would
    make the agent panic every time someone goes on holiday.

    The sample size is returned alongside the rate because the rate alone is
    misleading at low n: one skip out of two logged meals reads as 50%
    adherence, which is not evidence of a habit.
    """
    total = 0
    eaten = 0

    for log in logs:
        for meal in log.meals:
            if meal.status == MealStatus.PLANNED:
                continue
            total += 1
            if meal.status in (MealStatus.EATEN, MealStatus.SUBSTITUTED):
                eaten += 1

    if total == 0:
        return 1.0, 0  # no evidence of a problem
    return round(eaten / total, 3), total


def describe_snapshot(snapshot: AdherenceSnapshot) -> str:
    """Human-readable summary, used in prompts and the agent timeline."""
    parts = [
        f"{snapshot.meals_eaten}/{snapshot.meals_planned} meals eaten",
        f"{snapshot.calories_consumed}/{snapshot.calories_target} kcal",
        f"{snapshot.protein_consumed_g}/{snapshot.protein_target_g}g protein",
    ]
    if snapshot.meals_skipped:
        parts.append(f"{snapshot.meals_skipped} skipped today")
    if snapshot.skip_streak_days > 1:
        parts.append(f"{snapshot.skip_streak_days}-day skip streak")
    if snapshot.adherence_rate_7d < 1.0:
        parts.append(f"{snapshot.adherence_rate_7d:.0%} 7-day adherence")
    if snapshot.steps is not None:
        parts.append(f"{snapshot.steps} steps")
    return " · ".join(parts)
