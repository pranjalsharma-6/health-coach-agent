"""Output validation for generated plans.

Structured output guarantees a plan is *well-formed*. It says nothing about
whether the plan is *correct* — a schema-valid plan can still serve chicken to a
vegetarian, miss the protein target by 40g, or claim 50g of protein from a bowl
of rice.

This module is the second line of defence. Errors are blocking and trigger
regeneration; warnings are recorded but tolerated.
"""

import re
from dataclasses import dataclass, field
from math import ceil
from typing import List, Optional

from app.core.logging import get_logger
from app.models.enums import DietType
from app.models.plan import DailyPlan, HealthPlan, NutritionTargets
from app.models.profile import ProfileInDB
from app.services.ingredients import is_protein_claim_possible, protein_ceiling
from app.services.nutrition import (
    KCAL_PER_G_CARB,
    KCAL_PER_G_FAT,
    KCAL_PER_G_PROTEIN,
)

logger = get_logger(__name__)

# A plan shorter than this cannot rotate meaningfully — two days of food on
# repeat is not a plan, it is a pair of menus.
MIN_USABLE_PLAN_DAYS = 3

# How much shorter than requested a plan may be and still ship.
SHORT_PLAN_TOLERANCE = 0.7

# How far a day's totals may drift from target before it's a failure.
CALORIE_TOLERANCE = 0.12   # ±12%
PROTEIN_TOLERANCE = 0.15   # ±15%, and never more than 15% *under*

# A single meal shouldn't dominate the day.
MAX_SINGLE_MEAL_FRACTION = 0.55

# Meal macros should reconcile with its stated calories.
MACRO_RECONCILE_TOLERANCE = 0.30


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.is_valid:
            return "Plan passed validation."
        return "; ".join(self.errors)


def validate_plan(
    plan: HealthPlan,
    profile: ProfileInDB,
    targets: NutritionTargets,
    expected_days: Optional[int] = None,
) -> ValidationResult:
    """Run every check against a generated plan."""
    result = ValidationResult()

    _check_structure(plan, profile, result, expected_days)
    _check_diet_compliance(plan, profile, result)
    _check_allergens(plan, profile, result)

    diet = DietType(profile.diet_type)
    for day in plan.daily_plans:
        _check_day_totals(day, targets, result)
        _check_meal_plausibility(day, targets, diet, result)
        _check_training_session(day, profile, result)

    if result.errors:
        logger.warning(
            "Plan validation failed with %d error(s): %s",
            len(result.errors),
            result.summary(),
        )

    return result


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def _check_structure(
    plan: HealthPlan,
    profile: ProfileInDB,
    result: ValidationResult,
    expected_days: Optional[int] = None,
) -> None:
    if not plan.daily_plans:
        result.errors.append("Plan contains no days.")
        return

    # A short plan used to pass every check below — the numbering of [1, 2] is
    # perfectly sequential — so a two-day plan shipped as a week.
    #
    # But requiring the exact count throws away good plans. Models reliably
    # come up one day short on this kind of output, and a correct three-day
    # plan is worth far more to the user than a fourth attempt that also fails.
    # Plans record their own length and day resolution counts against the plan
    # rather than the request, so a short plan is coherent — it simply rotates
    # sooner. Only a plan too short to be a plan is rejected.
    if expected_days is not None:
        actual = len(plan.daily_plans)
        # Never demand more than was asked for: a one-day plan cannot be
        # too short when one day is the request.
        floor = min(
            expected_days,
            max(MIN_USABLE_PLAN_DAYS, ceil(expected_days * SHORT_PLAN_TOLERANCE)),
        )

        if actual > expected_days:
            result.errors.append(
                f"Plan has {actual} days, more than the {expected_days} requested."
            )
        elif actual < floor:
            result.errors.append(
                f"Plan has {actual} days, too few to be usable "
                f"(expected {expected_days})."
            )
        elif actual < expected_days:
            result.warnings.append(
                f"Plan has {actual} days rather than {expected_days}; it will "
                "rotate sooner."
            )

    day_numbers = [d.day for d in plan.daily_plans]
    expected = list(range(1, len(plan.daily_plans) + 1))
    if sorted(day_numbers) != expected:
        result.errors.append(
            f"Day numbering is wrong: got {sorted(day_numbers)}, expected {expected}."
        )

    for day in plan.daily_plans:
        if len(day.meals) != profile.meals_per_day:
            result.errors.append(
                f"Day {day.day} has {len(day.meals)} meals, "
                f"expected {profile.meals_per_day}."
            )

    seen_ids = set()
    for day in plan.daily_plans:
        for meal in day.meals:
            if meal.meal_id in seen_ids:
                result.errors.append(f"Duplicate meal_id '{meal.meal_id}'.")
            seen_ids.add(meal.meal_id)

    if not plan.agent_reasoning or len(plan.agent_reasoning.strip()) < 40:
        result.warnings.append("agent_reasoning is missing or too short to be useful.")


# --------------------------------------------------------------------------- #
# Diet compliance — the most important check
# --------------------------------------------------------------------------- #
def _searchable_text(day: DailyPlan) -> List[tuple[str, str]]:
    """Return (label, text) pairs to scan for forbidden ingredients."""
    out = []
    for meal in day.meals:
        text = f"{meal.name} {meal.description}"
        if meal.recipe:
            text += " " + " ".join(meal.recipe.ingredients)
        meal_type = getattr(meal.meal_type, "value", meal.meal_type)
        out.append((f"Day {day.day} {meal_type}", text.lower()))
    return out


def _contains_keyword(text: str, keyword: str) -> bool:
    """Whole-word match allowing only a plural suffix.

    The suffix has to be constrained: a trailing `\\w*` would catch "eggs" and
    "potatoes" but also fire "ham" on "hamper", which would reject a perfectly
    good halal plan.
    """
    return re.search(rf"\b{re.escape(keyword)}(?:s|es)?\b", text) is not None


def _check_diet_compliance(
    plan: HealthPlan, profile: ProfileInDB, result: ValidationResult
) -> None:
    diet = DietType(profile.diet_type)
    forbidden = diet.forbidden_keywords
    if not forbidden:
        return

    for day in plan.daily_plans:
        for label, text in _searchable_text(day):
            for keyword in forbidden:
                if _contains_keyword(text, keyword):
                    result.errors.append(
                        f"{label} contains '{keyword}', which is forbidden for a "
                        f"{diet.label} diet."
                    )


def _check_allergens(
    plan: HealthPlan, profile: ProfileInDB, result: ValidationResult
) -> None:
    if not profile.allergies:
        return

    for day in plan.daily_plans:
        for label, text in _searchable_text(day):
            for allergen in profile.allergies:
                if _contains_keyword(text, allergen):
                    result.errors.append(
                        f"{label} contains declared allergen '{allergen}'."
                    )


# --------------------------------------------------------------------------- #
# Nutrition accuracy
# --------------------------------------------------------------------------- #
def _check_day_totals(
    day: DailyPlan, targets: NutritionTargets, result: ValidationResult
) -> None:
    total_kcal = sum(m.calories_kcal for m in day.meals)
    total_protein = sum(m.protein_g for m in day.meals)

    kcal_low = targets.calories_kcal * (1 - CALORIE_TOLERANCE)
    kcal_high = targets.calories_kcal * (1 + CALORIE_TOLERANCE)

    if not kcal_low <= total_kcal <= kcal_high:
        result.errors.append(
            f"Day {day.day} totals {total_kcal} kcal, outside the acceptable range "
            f"{round(kcal_low)}-{round(kcal_high)} for a "
            f"{targets.calories_kcal} kcal target."
        )

    # Under-shooting protein is a real failure; overshooting it is fine.
    protein_floor = targets.protein_g * (1 - PROTEIN_TOLERANCE)
    if total_protein < protein_floor:
        result.errors.append(
            f"Day {day.day} provides only {total_protein}g protein, below the "
            f"{round(protein_floor)}g minimum for a {targets.protein_g}g target."
        )


def _check_meal_plausibility(
    day: DailyPlan,
    targets: NutritionTargets,
    diet: DietType,
    result: ValidationResult,
) -> None:
    """Catch physically impossible meals and lopsided days."""
    max_meal_kcal = targets.calories_kcal * MAX_SINGLE_MEAL_FRACTION

    for meal in day.meals:
        if meal.calories_kcal > max_meal_kcal:
            result.warnings.append(
                f"Day {day.day} {meal.meal_type} is {meal.calories_kcal} kcal — "
                f"over {int(MAX_SINGLE_MEAL_FRACTION * 100)}% of the day in one sitting."
            )

        if meal.calories_kcal <= 0:
            result.errors.append(
                f"Day {day.day} {meal.meal_type} has non-positive calories."
            )
            continue

        # Do the stated macros actually add up to the stated calories?
        derived = (
            meal.protein_g * KCAL_PER_G_PROTEIN
            + meal.carbs_g * KCAL_PER_G_CARB
            + meal.fat_g * KCAL_PER_G_FAT
        )
        drift = abs(derived - meal.calories_kcal) / meal.calories_kcal

        if drift > MACRO_RECONCILE_TOLERANCE:
            result.errors.append(
                f"Day {day.day} {meal.meal_type} ('{meal.name}') claims "
                f"{meal.calories_kcal} kcal but its macros total {derived} kcal — "
                "these are inconsistent."
            )

        # Reconciliation alone can't catch this: 50g protein with no carbs or
        # fat in a 200 kcal meal adds up perfectly and is still impossible.
        # No food available on this diet is that protein-dense.
        if not is_protein_claim_possible(meal.calories_kcal, meal.protein_g, diet):
            density = meal.protein_g / meal.calories_kcal
            result.errors.append(
                f"Day {day.day} {meal.meal_type} ('{meal.name}') claims "
                f"{meal.protein_g}g protein in {meal.calories_kcal} kcal "
                f"({density:.2f} g/kcal). Nothing available on a {diet.label} "
                f"diet exceeds {protein_ceiling(diet):.2f} g/kcal, so this meal "
                "cannot exist as described."
            )


def build_retry_feedback(result: ValidationResult) -> str:
    """Turn validation errors into a correction instruction for the model.

    Naming the specific failures produces far better second attempts than a bare
    'that was wrong, try again'.
    """
    lines = [
        "Your previous plan was REJECTED by automated validation.",
        "",
        "Problems found:",
    ]
    lines.extend(f"{i}. {err}" for i, err in enumerate(result.errors, start=1))
    lines.extend(
        [
            "",
            "Regenerate the ENTIRE plan, fixing every problem above. Pay particular "
            "attention to diet restrictions and to making each day's calorie and "
            "protein totals land within tolerance of the targets.",
        ]
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _check_training_session(
    day: DailyPlan, profile: ProfileInDB, result: ValidationResult
) -> None:
    """Is this session something the user can actually do?

    The same division as everywhere else: the model chooses the movements, and
    deterministic code checks the choice is feasible. What counts as good
    programming is the model's to argue; what counts as *followable* is not.

    A missing session is an error rather than a warning — "Strength training,
    45 min" with no exercises is the exact failure this exists to stop, and it
    was the shipped behaviour.
    """
    from app.agent.prompts import training_level_for
    from app.services.exercises import find_problems

    activity = day.activity
    is_rest = activity.duration_minutes == 0 or "rest" in activity.activity_type.lower()

    if is_rest:
        if activity.exercises:
            result.warnings.append(
                f"Day {day.day} is a rest day but lists exercises."
            )
        return

    if not activity.exercises:
        result.errors.append(
            f"Day {day.day} prescribes '{activity.activity_type}' with no "
            "exercises. A category is not a session someone can follow."
        )
        return

    problems = find_problems(
        [e.name for e in activity.exercises],
        training_level_for(profile),
        profile.training_styles,
    )
    for problem in problems:
        result.errors.append(f"Day {day.day}: {problem}")
