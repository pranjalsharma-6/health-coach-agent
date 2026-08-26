"""Health profile — everything captured during onboarding.

Every field here is a hard constraint on plan generation, not decoration.
If a field can't change the output, it shouldn't be in the onboarding flow.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.models.common import MongoModel, utcnow
from app.models.enums import (
    ActivityLevel,
    BudgetTier,
    CookingSkill,
    Cuisine,
    DietType,
    Gender,
    Goal,
)


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
    target_timeline_weeks: int = Field(default=12, ge=4, le=52)

    # --- Diet (the adherence levers) ---
    diet_type: DietType
    cuisine_preference: Cuisine = Cuisine.MIXED
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
    """Partial update — every field optional."""

    gender: Optional[Gender] = None
    age_years: Optional[int] = Field(default=None, ge=13, le=100)
    height_cm: Optional[float] = Field(default=None, ge=100, le=250)
    current_weight_kg: Optional[float] = Field(default=None, ge=30, le=300)
    target_weight_kg: Optional[float] = Field(default=None, ge=30, le=300)
    goal: Optional[Goal] = None
    activity_level: Optional[ActivityLevel] = None
    target_timeline_weeks: Optional[int] = Field(default=None, ge=4, le=52)
    diet_type: Optional[DietType] = None
    cuisine_preference: Optional[Cuisine] = None
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
