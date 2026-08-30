"""Deterministic nutrition math.

Nothing in this module calls an LLM. Every number a user's health depends on is
computed here, from published equations, in code that can be unit-tested.

The LLM is allowed to choose *what food*, never *how many calories*.
"""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.models.enums import ActivityLevel, Gender, Goal
from app.models.plan import NutritionTargets
from app.models.profile import ProfileBase

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Safety floors. These are hard limits, not suggestions.
# --------------------------------------------------------------------------- #
ABSOLUTE_MIN_KCAL_FEMALE = 1200
ABSOLUTE_MIN_KCAL_MALE = 1500
MAX_DAILY_DEFICIT_KCAL = 1000
MAX_DAILY_SURPLUS_KCAL = 500

# Protein targets in g per kg of bodyweight.
PROTEIN_G_PER_KG = {
    Goal.FAT_LOSS: 2.0,        # high, to preserve lean mass in a deficit
    Goal.MUSCLE_GAIN: 1.8,
    Goal.MAINTENANCE: 1.4,
    Goal.ENDURANCE: 1.6,
    Goal.GENERAL_HEALTH: 1.2,
}

# Minimum dietary fat, for hormone function and fat-soluble vitamin absorption.
MIN_FAT_G_PER_KG = 0.6

KCAL_PER_G_PROTEIN = 4
KCAL_PER_G_CARB = 4
KCAL_PER_G_FAT = 9


@dataclass(frozen=True)
class EnergyProfile:
    """Computed energy figures for a user."""

    bmr_kcal: int
    tdee_kcal: int
    target_kcal: int
    deficit_or_surplus_kcal: int
    safety_floor_applied: bool


def calculate_bmr(
    weight_kg: float, height_cm: float, age_years: int, gender: Gender
) -> float:
    """Basal metabolic rate via the Mifflin-St Jeor equation.

    Mifflin-St Jeor is the current standard. It outperforms Harris-Benedict on
    modern populations. For `OTHER`, we average the male and female constants
    rather than defaulting to one, which would systematically misestimate.
    """
    base = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years)

    if gender == Gender.MALE:
        return base + 5
    if gender == Gender.FEMALE:
        return base - 161
    return base + ((5 + -161) / 2)  # -78, the midpoint


def calculate_tdee(bmr: float, profile: ProfileBase) -> float:
    """Total daily energy expenditure = BMR x activity multiplier.

    `activity_level` arrives as a plain string on models loaded from Mongo
    (`use_enum_values=True`) and as an enum on freshly-validated input, so
    coerce before reaching for enum members.
    """
    return bmr * ActivityLevel(profile.activity_level).multiplier


def calculate_targets(profile: ProfileBase) -> NutritionTargets:
    """Derive daily calorie and macro targets from a profile.

    Order matters: protein and fat floors are satisfied first, and carbohydrate
    absorbs the remainder. That way a low-calorie target can never squeeze
    protein below the level needed to preserve lean mass.
    """
    energy = calculate_energy_profile(profile)
    weight = profile.current_weight_kg

    # Protein: goal-driven, floored so it can never collapse.
    protein_g = round(weight * PROTEIN_G_PER_KG[Goal(profile.goal)])
    protein_kcal = protein_g * KCAL_PER_G_PROTEIN

    # Fat: at least the essential minimum, otherwise ~25% of intake.
    min_fat_g = round(weight * MIN_FAT_G_PER_KG)
    preferred_fat_g = round((energy.target_kcal * 0.25) / KCAL_PER_G_FAT)
    fat_g = max(min_fat_g, preferred_fat_g)
    fat_kcal = fat_g * KCAL_PER_G_FAT

    # Carbs: whatever energy is left.
    carb_kcal = max(energy.target_kcal - protein_kcal - fat_kcal, 0)
    carbs_g = round(carb_kcal / KCAL_PER_G_CARB)

    if carb_kcal == 0:
        logger.warning(
            "Protein and fat floors consume the entire calorie target for user "
            "profile (target=%s kcal). Carbohydrate set to zero.",
            energy.target_kcal,
        )

    # Water: 35 ml/kg is the common clinical rule of thumb.
    water_ml = int(round(weight * 35 / 50) * 50)

    return NutritionTargets(
        calories_kcal=energy.target_kcal,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        water_ml=water_ml,
    )


def calculate_energy_profile(profile: ProfileBase) -> EnergyProfile:
    """Compute BMR, TDEE and the safe calorie target for the user's goal."""
    bmr = calculate_bmr(
        profile.current_weight_kg,
        profile.height_cm,
        profile.age_years,
        Gender(profile.gender),
    )
    tdee = calculate_tdee(bmr, profile)

    adjustment = _goal_adjustment(profile, tdee)
    raw_target = tdee + adjustment

    floor = (
        ABSOLUTE_MIN_KCAL_FEMALE
        if Gender(profile.gender) == Gender.FEMALE
        else ABSOLUTE_MIN_KCAL_MALE
    )
    target = max(raw_target, floor)
    floor_applied = target > raw_target

    if floor_applied:
        logger.info(
            "Calorie floor applied: requested %s kcal, raised to %s kcal.",
            round(raw_target),
            target,
        )

    return EnergyProfile(
        bmr_kcal=round(bmr),
        tdee_kcal=round(tdee),
        target_kcal=round(target),
        deficit_or_surplus_kcal=round(target - tdee),
        safety_floor_applied=floor_applied,
    )


def _goal_adjustment(profile: ProfileBase, tdee: float) -> float:
    """Calorie delta from maintenance, clamped to a safe rate of change.

    For fat loss we size the deficit from the user's actual timeline rather than
    hardcoding 500 kcal, then clamp it, so an over-ambitious timeline produces a
    slower plan instead of an unsafe one.
    """
    goal = Goal(profile.goal)

    if goal == Goal.FAT_LOSS:
        if profile.target_weight_kg is None:
            return -500.0
        kg_to_lose = max(profile.current_weight_kg - profile.target_weight_kg, 0)
        # ~7,700 kcal per kg of fat tissue.
        total_deficit = kg_to_lose * 7700
        days = max(profile.target_timeline_weeks * 7, 1)
        required = total_deficit / days
        return -min(required, MAX_DAILY_DEFICIT_KCAL)

    if goal == Goal.MUSCLE_GAIN:
        return min(300.0, MAX_DAILY_SURPLUS_KCAL)

    if goal == Goal.ENDURANCE:
        return 150.0

    return 0.0  # maintenance, general health


def estimated_weekly_change_kg(profile: ProfileBase) -> float:
    """Projected weight change per week at the current target. Signed."""
    energy = calculate_energy_profile(profile)
    return round((energy.deficit_or_surplus_kcal * 7) / 7700, 2)


def macro_split_percent(targets: NutritionTargets) -> dict[str, int]:
    """Macro breakdown as percentages, for the UI's donut chart."""
    protein_kcal = targets.protein_g * KCAL_PER_G_PROTEIN
    carb_kcal = targets.carbs_g * KCAL_PER_G_CARB
    fat_kcal = targets.fat_g * KCAL_PER_G_FAT
    total = protein_kcal + carb_kcal + fat_kcal

    if total == 0:
        return {"protein": 0, "carbs": 0, "fat": 0}

    return {
        "protein": round(protein_kcal / total * 100),
        "carbs": round(carb_kcal / total * 100),
        "fat": round(fat_kcal / total * 100),
    }
