"""Health profile. Everything captured during onboarding.

Every field here is a hard constraint on plan generation, not decoration.
If a field can't change the output, it shouldn't be in the onboarding flow.
"""

from datetime import datetime
from typing import Annotated, Any, List, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    Field,
    field_validator,
)

from app.models.common import MongoModel, utcnow
from app.models.enums import (
    ActivityLevel,
    BudgetTier,
    CookingSkill,
    Cuisine,
    DietType,
    Gender,
    Goal,
    TrainingStyle,
)


def _coerce_cuisines(value: Any) -> Any:
    """Accept the old single value as well as a list.

    This was one enum until the obvious objection landed: someone who eats
    North Indian on weekdays and continental at the weekend is not an edge
    case. Profiles already in Mongo hold a bare string, and rejecting them
    would lock those users out of their own plan, so the old shape is read as
    a one-item list rather than migrated.

    An empty selection means "no strong preference", which is what MIXED
    already encodes. Mapping it there beats leaving the planner with no
    cuisine guidance at all.
    """
    if value is None:
        return value
    if isinstance(value, (str, Cuisine)):
        return [value]
    if isinstance(value, list) and not value:
        return [Cuisine.MIXED]
    return value


CuisineList = Annotated[List[Cuisine], BeforeValidator(_coerce_cuisines)]


def _coerce_styles(value: Any) -> Any:
    """Same shape as cuisines: a bare value becomes a list, empty means default.

    An empty selection is not an error. It means "no preference", and
    bodyweight is the honest default because it needs nothing and excludes
    nobody.
    """
    if value is None:
        return value
    if isinstance(value, (str, TrainingStyle)):
        return [value]
    if isinstance(value, list) and not value:
        return [TrainingStyle.BODYWEIGHT]
    return value


TrainingStyleList = Annotated[List[TrainingStyle], BeforeValidator(_coerce_styles)]


class ProfileBase(BaseModel):
    # --- Body ---
    gender: Gender
    age_years: int = Field(ge=13, le=100)
    height_cm: float = Field(ge=100, le=250)
    current_weight_kg: float = Field(ge=30, le=300)
    target_weight_kg: Optional[float] = Field(default=None, ge=30, le=300)

    # --- Goal ---
    goal: Goal
    activity_level: ActivityLevel = ActivityLevel.MODERATELY_ACTIVE
    # What the trainer may prescribe from. Defaults to bodyweight so a profile
    # written before this existed still plans something the user can do.
    training_styles: TrainingStyleList = Field(
        default_factory=lambda: [TrainingStyle.BODYWEIGHT]
    )
    target_timeline_weeks: int = Field(default=12, ge=4, le=52)

    # --- Diet (the adherence levers) ---
    diet_type: DietType
    # The alias reads profiles written before this was a list. Without it a
    # stored `cuisine_preference` is simply an unknown key, and the user's
    # choice silently resets to "mixed" the next time they load their profile.
    cuisine_preferences: CuisineList = Field(
        default_factory=lambda: [Cuisine.MIXED],
        validation_alias=AliasChoices("cuisine_preferences", "cuisine_preference"),
    )
    allergies: List[str] = Field(default_factory=list)
    disliked_foods: List[str] = Field(default_factory=list)

    # --- Practical constraints ---
    meals_per_day: int = Field(default=4, ge=2, le=6)
    cooking_skill: CookingSkill = CookingSkill.BEGINNER
    max_prep_minutes: int = Field(default=30, ge=5, le=180)
    budget_tier: BudgetTier = BudgetTier.MEDIUM
    eat_out_per_week: int = Field(default=2, ge=0, le=21)

    # --- Safety ---
    medical_notes: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("allergies", "disliked_foods")
    @classmethod
    def _clean_list(cls, v: List[str]) -> List[str]:
        """Normalise to lowercase, drop blanks and duplicates, cap the length."""
        seen, out = set(), []
        for item in v:
            cleaned = item.strip().lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
        return out[:25]

    @property
    def bmi(self) -> float:
        height_m = self.height_cm / 100
        return round(self.current_weight_kg / (height_m**2), 1)


class ProfileCreate(ProfileBase):
    """Payload submitted by the onboarding wizard."""


class ProfileUpdate(BaseModel):
    """Partial update. Every field optional."""

    gender: Optional[Gender] = None
    age_years: Optional[int] = Field(default=None, ge=13, le=100)
    height_cm: Optional[float] = Field(default=None, ge=100, le=250)
    current_weight_kg: Optional[float] = Field(default=None, ge=30, le=300)
    target_weight_kg: Optional[float] = Field(default=None, ge=30, le=300)
    goal: Optional[Goal] = None
    activity_level: Optional[ActivityLevel] = None
    training_styles: Optional[TrainingStyleList] = None
    target_timeline_weeks: Optional[int] = Field(default=None, ge=4, le=52)
    diet_type: Optional[DietType] = None
    cuisine_preferences: Optional[CuisineList] = Field(
        default=None,
        validation_alias=AliasChoices(
            "cuisine_preferences", "cuisine_preference"
        ),
    )
    allergies: Optional[List[str]] = None
    disliked_foods: Optional[List[str]] = None
    meals_per_day: Optional[int] = Field(default=None, ge=2, le=6)
    cooking_skill: Optional[CookingSkill] = None
    max_prep_minutes: Optional[int] = Field(default=None, ge=5, le=180)
    budget_tier: Optional[BudgetTier] = None
    eat_out_per_week: Optional[int] = Field(default=None, ge=0, le=21)
    medical_notes: Optional[str] = Field(default=None, max_length=1000)


class ProfileInDB(MongoModel, ProfileBase):
    user_id: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
