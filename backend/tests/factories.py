"""Test fixtures shared across agent tests."""

from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from app.models.enums import (
    ActivityLevel,
    DietType,
    Gender,
    Goal,
    MealStatus,
    MealType,
)
from app.models.log import DailyLogInDB, MealLogEntry
from app.models.plan import (
    ActivityItem,
    DailyPlan,
    HealthPlan,
    MealItem,
    NutritionTargets,
    PlanInDB,
)
from app.models.profile import ProfileInDB


def make_profile(**overrides) -> ProfileInDB:
    defaults = dict(
        user_id="test-user",
        gender=Gender.MALE,
        age_years=30,
        height_cm=175,
        current_weight_kg=85.0,
        target_weight_kg=75.0,
        goal=Goal.FAT_LOSS,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        target_timeline_weeks=12,
        diet_type=DietType.VEGETARIAN,
        meals_per_day=4,
    )
    defaults.update(overrides)
    return ProfileInDB(**defaults)


def make_targets(calories: int = 2000, protein: int = 170) -> NutritionTargets:
    # Fat at 25% of calories, carbs absorb the remainder. Mirrors the real engine.
    fat_g = round(calories * 0.25 / 9)
    carbs_g = round((calories - protein * 4 - fat_g * 9) / 4)
    return NutritionTargets(
        calories_kcal=calories, protein_g=protein, carbs_g=carbs_g, fat_g=fat_g
    )


def make_meal(
    meal_id: str,
    meal_type: MealType = MealType.BREAKFAST,
    # Deliberately free of any keyword the diet or allergen scans look for,
    # so a clean fixture stays clean.
    name: str = "Masala oats with mixed vegetables",
    description: str = "Savoury oats simmered with seasonal vegetables.",
    calories: int = 500,
    protein: int = 42,
) -> MealItem:
    """Build a meal whose macros reconcile with its calories.

    Protein and fat are set first, carbs absorb the remainder, so the
    validator's reconciliation check passes for well-formed fixtures.
    """
    fat_g = round(calories * 0.25 / 9)
    carbs_kcal = max(calories - protein * 4 - fat_g * 9, 0)
    return MealItem(
        meal_id=meal_id,
        meal_type=meal_type,
        name=name,
        description=description,
        calories_kcal=calories,
        protein_g=protein,
        carbs_g=round(carbs_kcal / 4),
        fat_g=fat_g,
    )


# A valid beginner session: four patterns, no equipment at all, every movement
# in the table. Equipment-free on purpose. Bodyweight is the default training
# style, so this fixture stays valid for a profile that made no choices.
def make_exercises(draft: bool = False) -> list:
    """`draft=True` returns the trainer's output shape, which carries no cue,
    that is filled from the exercise table during assembly."""
    from app.models.plan import ExerciseDraft, ExercisePrescription

    cls = ExerciseDraft if draft else ExercisePrescription
    return [
        cls(name="Push-ups", sets=3, reps="8-12", rest_seconds=90),
        cls(name="Inverted row", sets=3, reps="8-12", rest_seconds=90),
        cls(name="Bodyweight squat", sets=3, reps="12-20", rest_seconds=90),
        cls(name="Plank", sets=3, reps="30 seconds", rest_seconds=60),
    ]


def make_cardio_exercises(draft: bool = False) -> list:
    from app.models.plan import ExerciseDraft, ExercisePrescription

    cls = ExerciseDraft if draft else ExercisePrescription
    return [cls(name="Brisk walk", sets=1, reps="30 minutes", rest_seconds=0)]


def make_day(day: int, targets: NutritionTargets, meals_per_day: int = 4) -> DailyPlan:
    """Build a day whose totals land on the targets."""
    slots = [MealType.BREAKFAST, MealType.LUNCH, MealType.DINNER, MealType.SNACK]
    per_meal_kcal = targets.calories_kcal // meals_per_day
    per_meal_protein = targets.protein_g // meals_per_day

    # Push rounding remainder into the first meal so the day totals exactly.
    kcal_remainder = targets.calories_kcal - per_meal_kcal * meals_per_day
    protein_remainder = targets.protein_g - per_meal_protein * meals_per_day

    meals = []
    for i in range(meals_per_day):
        slot = slots[i] if i < len(slots) else MealType.SNACK
        meals.append(
            make_meal(
                meal_id=f"d{day}-{slot.value}" + (f"-{i}" if i >= len(slots) else ""),
                meal_type=slot,
                calories=per_meal_kcal + (kcal_remainder if i == 0 else 0),
                protein=per_meal_protein + (protein_remainder if i == 0 else 0),
            )
        )

    return DailyPlan(
        day=day,
        meals=meals,
        activity=ActivityItem(
            activity_type="Strength training. Full body",
            duration_minutes=45,
            intensity="moderate",
            description="Compound lifts, focus on form.",
            target_steps=8000,
            exercises=make_exercises(),
        ),
    )


def make_health_plan(
    targets: NutritionTargets, days: int = 7, meals_per_day: int = 4
) -> HealthPlan:
    return HealthPlan(
        plan_title="Week 1: Protein First",
        duration_days=days,
        agent_reasoning=(
            "Built around vegetarian protein anchors because hitting 170g without "
            "meat needs deliberate planning. Meals are simple and repeat across the "
            "week to keep prep under 30 minutes."
        ),
        daily_plans=[make_day(d, targets, meals_per_day) for d in range(1, days + 1)],
    )


def make_plan_in_db(
    targets: NutritionTargets,
    created_days_ago: int = 0,
    duration_days: int = 7,
    meals_per_day: int = 4,
    reference_date: Optional[date] = None,
) -> PlanInDB:
    """Build a stored plan.

    `reference_date` anchors `created_at` to the test's notion of today rather
    than the wall clock, so tests that use a fixed date aren't silently broken
    by the real calendar.
    """
    anchor = reference_date or date.today()
    created_at = datetime.combine(
        anchor - timedelta(days=created_days_ago),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )

    generated = make_health_plan(targets, duration_days, meals_per_day)
    return PlanInDB(
        _id="plan-1",
        user_id="test-user",
        plan_title=generated.plan_title,
        duration_days=duration_days,
        agent_reasoning=generated.agent_reasoning,
        daily_plans=generated.daily_plans,
        targets=targets,
        created_at=created_at,
    )


def make_log(
    log_date: date,
    statuses: Optional[List[tuple[str, MealStatus]]] = None,
    **metrics,
) -> DailyLogInDB:
    """Build a daily log from (meal_id, status) pairs."""
    entries = [
        MealLogEntry(meal_id=meal_id, status=status)
        for meal_id, status in (statuses or [])
    ]
    return DailyLogInDB(
        user_id="test-user", log_date=log_date, meals=entries, **metrics
    )


# --------------------------------------------------------------------------- #
# Specialist drafts
# --------------------------------------------------------------------------- #
def make_meal_draft(
    targets: NutritionTargets, days: int = 7, meals_per_day: int = 4
) -> "MealPlanDraft":
    """The nutritionist's half of a well-formed plan."""
    from app.models.plan import DayMeals, MealDraftItem, MealPlanDraft

    return MealPlanDraft(
        plan_title="Week 1: Protein First",
        reasoning=(
            "Built around vegetarian protein anchors because hitting the target "
            "without meat needs deliberate planning. Meals repeat across the week "
            "to keep prep under 30 minutes."
        ),
        days=[
            DayMeals(
                day=d,
                meals=[
                    MealDraftItem.from_meal_item(m)
                    for m in make_day(d, targets, meals_per_day).meals
                ],
            )
            for d in range(1, days + 1)
        ],
    )


def make_training_draft(days: int = 7) -> "TrainingPlanDraft":
    """The trainer's half. A sane week with one genuine rest day."""
    from app.models.plan import DayTraining, TrainingPlanDraft

    def activity(day: int) -> "ActivityDraft":
        from app.models.plan import ActivityDraft

        if day == 4:
            return ActivityDraft(
                activity_type="Rest",
                duration_minutes=0,
                intensity="low",
                description="Full recovery day.",
            )
        hard = bool(day % 2)
        return ActivityDraft(
            activity_type="Strength training. Full body" if hard else "Brisk walk",
            duration_minutes=45 if hard else 30,
            intensity="moderate" if hard else "low",
            description=(
                "Compound lifts, focus on form."
                if hard
                else "Easy pace, keep it conversational."
            ),
            exercises=(
                make_exercises(draft=True)
                if hard
                else make_cardio_exercises(draft=True)
            ),
        )

    return TrainingPlanDraft(
        reasoning=(
            "Alternating strength and easy movement, with a genuine rest day "
            "mid-week so the deficit doesn't compound into fatigue."
        ),
        days=[DayTraining(day=d, activity=activity(d)) for d in range(1, days + 1)],
    )


def make_critique(approved: bool = True, issues=None) -> "PlanCritique":
    from app.models.plan import PlanCritique

    issues = list(issues or [])
    return PlanCritique(
        approved=approved and not issues,
        issues=issues,
        summary=(
            "The week is coherent and the training fits the deficit."
            if approved and not issues
            else "The week needs a few adjustments before it's ready."
        ),
    )


def scope_to_requested_days(draft, messages):
    """Trim a meal draft to the day range the prompt asked for.

    The nutritionist drafts the week in chunks, so a stub that returns all
    seven days for every chunk stitches into a fourteen-day plan and hides
    whatever the test was actually checking. Real models return what they were
    asked for; stubs should too.
    """
    import re

    text = (
        messages
        if isinstance(messages, str)
        else "\n".join(str(getattr(m, "content", m)) for m in messages)
    )
    match = re.search(r"DAYS (\d+) TO (\d+) ONLY", text)
    if not match:
        return draft

    first, last = int(match.group(1)), int(match.group(2))
    return draft.model_copy(
        update={"days": [d for d in draft.days if first <= d.day <= last]}
    )
